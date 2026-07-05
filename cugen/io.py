"""cugen.io — .cugen file I/O for cugen.

Self-contained port of cugen_reader.py + cugen_reader_pinned.py with no
runtime import of the original scripts. Public API:

    CugenReader            mmap-backed reader (default)
    CugenReaderPinned      pinned-host + async-H2D variant (faster sequential)
    read_cugen_header(path) -> dict
    read_cugen(path)        -> CugenReader (or pinned, if USE_PINNED_READER=1)
    write_cugen()           NotImplementedError (v0.2)

Cugen binary format (legacy magic 'CUPGEN01' preserved post-rename):
    256-byte header: magic, version, encoding, n_samples, n_variants,
                     bytes_per_variant, stats_offset, data_offset,
                     gidx_offset, flags
    Stats block:     3 * n_variants * float32 (mu_x, sxx, maf)
    Data block:      n_variants * bytes_per_variant (packed 2-bit)
    Optional gidx:   n_variants * int64

Genotype encoding (2-bit, big-endian within byte):
    0,1,2 = dosage; 3 = missing (treated as 0 after centering)
"""

import mmap
import os
import struct
from typing import Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

CUGEN_MAGIC = b"CUPGEN01"
CUGEN_VERSION = 1
HEADER_SIZE = 256

ENCODING_2BIT = 0
ENCODING_UINT8 = 1
ENCODING_FLOAT16 = 2
ENCODING_FLOAT32 = 3

FLAG_HAS_MISSING = 1
FLAG_HAS_GIDX_MAP = 2

_UNPACK_2BIT_KERNEL = None
_FUSED_UNIVARIATE_KERNEL = None


def _get_unpack_kernel():
    global _UNPACK_2BIT_KERNEL
    if _UNPACK_2BIT_KERNEL is None and HAS_CUPY:
        _UNPACK_2BIT_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void unpack_2bit_to_float32(
            const unsigned char* packed,
            float* output,
            const long long n_samples,
            const long long n_variants,
            const long long bytes_per_variant
        ) {
            long long sample_idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (sample_idx >= n_samples) return;
            long long byte_idx = sample_idx / 4;
            int bit_shift = 6 - 2 * (sample_idx % 4);
            for (long long var_idx = 0; var_idx < n_variants; var_idx++) {
                long long packed_offset = var_idx * bytes_per_variant + byte_idx;
                unsigned char byte_val = packed[packed_offset];
                unsigned char geno = (byte_val >> bit_shift) & 0x03;
                float val = (geno == 3) ? 0.0f : (float)geno;
                output[sample_idx * n_variants + var_idx] = val;
            }
        }
        ''', 'unpack_2bit_to_float32')
    return _UNPACK_2BIT_KERNEL


def _get_fused_univariate_kernel():
    global _FUSED_UNIVARIATE_KERNEL
    if _FUSED_UNIVARIATE_KERNEL is None and HAS_CUPY:
        _FUSED_UNIVARIATE_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void fused_univariate(
            const unsigned char* packed,
            const float* y_centered,
            const float* mu_x,
            const float* sxx,
            float* num_out,
            const long long n_samples,
            const long long n_variants,
            const long long bytes_per_variant,
            const float lambda_reg
        ) {
            long long var_idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (var_idx >= n_variants) return;
            float mu = mu_x[var_idx];
            double num = 0.0;
            for (long long s = 0; s < n_samples; s++) {
                long long byte_idx = s / 4;
                int bit_shift = 6 - 2 * (s % 4);
                long long packed_offset = var_idx * bytes_per_variant + byte_idx;
                unsigned char byte_val = packed[packed_offset];
                unsigned char geno = (byte_val >> bit_shift) & 0x03;
                if (geno == 3) continue;   // complete-case: EXCLUDE missing from the
                                           // numerator (do NOT treat as dosage 0 — that
                                           // adds spurious (0-mu)*y variance and inflates
                                           // Z for high-missing/common-ALT variants).
                                           // Matches the stored non-missing sxx. (session 52)
                float x = (float)geno;
                num += (double)(x - mu) * (double)y_centered[s];
            }
            num_out[var_idx] = (float)num;
        }
        ''', 'fused_univariate')
    return _FUSED_UNIVARIATE_KERNEL


class CugenReader:
    """mmap-backed reader for .cugen files.

    Stats (mu_x, sxx, maf) are loaded into RAM on open — small (3 * n_variants
    floats) and frequently accessed. Genotype block stays mmap'd.
    """

    def __init__(self, path: str, device: int = 0, use_gds: bool = False):
        self.path = path
        self.device = device
        self.use_gds = False  # GDS unreachable on Sherlock, kept for API compat

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'rb') as f:
            header = f.read(HEADER_SIZE)
        self._parse_header(header)

        self._file = open(path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._load_stats()

        if HAS_CUPY:
            cp.cuda.Device(device).use()

    def _parse_header(self, header: bytes):
        magic = header[0:8]
        if magic != CUGEN_MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r} (expected {CUGEN_MAGIC!r})")
        self.version = struct.unpack_from('<I', header, 8)[0]
        self.encoding = struct.unpack_from('<I', header, 12)[0]
        self.n_samples = struct.unpack_from('<Q', header, 16)[0]
        self.n_variants = struct.unpack_from('<Q', header, 24)[0]
        self.bytes_per_variant = struct.unpack_from('<Q', header, 32)[0]
        self.stats_offset = struct.unpack_from('<Q', header, 40)[0]
        self.data_offset = struct.unpack_from('<Q', header, 48)[0]
        self.gidx_offset = struct.unpack_from('<Q', header, 56)[0]
        self.flags = struct.unpack_from('<I', header, 64)[0]
        self.has_missing = bool(self.flags & FLAG_HAS_MISSING)
        self.has_gidx_map = bool(self.flags & FLAG_HAS_GIDX_MAP)

    def _load_stats(self):
        n = self.n_variants
        self._mmap.seek(self.stats_offset)
        self.mu_x = np.frombuffer(self._mmap.read(n * 4), dtype=np.float32).copy()
        self.sxx = np.frombuffer(self._mmap.read(n * 4), dtype=np.float32).copy()
        self.maf = np.frombuffer(self._mmap.read(n * 4), dtype=np.float32).copy()
        if self.has_gidx_map:
            self._mmap.seek(self.gidx_offset)
            self.gidx = np.frombuffer(self._mmap.read(n * 8), dtype=np.int64).copy()
        else:
            self.gidx = np.arange(n, dtype=np.int64)

    def close(self):
        if hasattr(self, '_mmap') and self._mmap:
            self._mmap.close()
        if hasattr(self, '_file') and self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return self.n_variants

    @property
    def shape(self):
        return (self.n_samples, self.n_variants)

    def get_stats(self, start: int = 0, end: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if end is None:
            end = self.n_variants
        return self.mu_x[start:end], self.sxx[start:end], self.maf[start:end]

    def get_gidx(self, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        if end is None:
            end = self.n_variants
        return self.gidx[start:end]

    def read_packed_bytes(self, start: int = 0, end: Optional[int] = None) -> bytes:
        if end is None:
            end = self.n_variants
        n_variants = end - start
        byte_offset = self.data_offset + start * self.bytes_per_variant
        n_bytes = n_variants * self.bytes_per_variant
        self._mmap.seek(byte_offset)
        return self._mmap.read(n_bytes)

    def read_indices_to_gpu(self, indices: Union[list, np.ndarray],
                            stream: Optional['cp.cuda.Stream'] = None) -> 'cp.ndarray':
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")
        indices = np.asarray(indices, dtype=np.int64)
        n_variants = len(indices)
        if n_variants == 0:
            return cp.empty((self.n_samples, 0), dtype=cp.float32)

        sort_order = np.argsort(indices)
        sorted_indices = indices[sort_order]
        cp.cuda.Device(self.device).use()

        packed_buffer = np.empty(n_variants * self.bytes_per_variant, dtype=np.uint8)
        for i, var_idx in enumerate(sorted_indices):
            byte_offset = self.data_offset + var_idx * self.bytes_per_variant
            self._mmap.seek(byte_offset)
            s = i * self.bytes_per_variant
            packed_buffer[s:s + self.bytes_per_variant] = np.frombuffer(
                self._mmap.read(self.bytes_per_variant), dtype=np.uint8
            )

        packed_gpu = cp.asarray(packed_buffer)
        X_sorted = cp.empty((self.n_samples, n_variants), dtype=cp.float32)
        kernel = _get_unpack_kernel()
        threads_per_block = 256
        blocks = (self.n_samples + threads_per_block - 1) // threads_per_block
        kernel((blocks,), (threads_per_block,), (
            packed_gpu, X_sorted,
            np.int64(self.n_samples), np.int64(n_variants),
            np.int64(self.bytes_per_variant)
        ))
        if stream:
            stream.synchronize()
        else:
            cp.cuda.Device().synchronize()

        inverse_order = np.argsort(sort_order)
        X_gpu = X_sorted[:, cp.asarray(inverse_order)]
        del packed_gpu, X_sorted
        cp.get_default_memory_pool().free_all_blocks()
        return X_gpu

    def read_indices_to_gpu_batched(self, indices: Union[list, np.ndarray],
                                    stream: Optional['cp.cuda.Stream'] = None) -> 'cp.ndarray':
        """Like read_indices_to_gpu but coalesces contiguous index runs into single mmap reads.

        Reduces syscall count 10-100× when adjacent indices survive top-K selection.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")
        indices = np.asarray(indices, dtype=np.int64)
        n_variants = len(indices)
        if n_variants == 0:
            return cp.empty((self.n_samples, 0), dtype=cp.float32)

        sort_order = np.argsort(indices)
        sorted_indices = indices[sort_order]
        cp.cuda.Device(self.device).use()

        bpv = self.bytes_per_variant
        packed_buffer = np.empty(n_variants * bpv, dtype=np.uint8)

        diffs = np.diff(sorted_indices)
        run_breaks = np.where(diffs != 1)[0]
        run_start_pos = np.concatenate([[0], run_breaks + 1])
        run_end_pos = np.concatenate([run_breaks, [n_variants - 1]])

        for rs, re_ in zip(run_start_pos, run_end_pos):
            run_len = int(re_) - int(rs) + 1
            first_var_idx = int(sorted_indices[rs])
            byte_offset = self.data_offset + first_var_idx * bpv
            run_nbytes = run_len * bpv
            self._mmap.seek(byte_offset)
            buf_s = int(rs) * bpv
            packed_buffer[buf_s:buf_s + run_nbytes] = np.frombuffer(
                self._mmap.read(run_nbytes), dtype=np.uint8
            )

        packed_gpu = cp.asarray(packed_buffer)
        X_sorted = cp.empty((self.n_samples, n_variants), dtype=cp.float32)
        kernel = _get_unpack_kernel()
        threads_per_block = 256
        blocks = (self.n_samples + threads_per_block - 1) // threads_per_block
        kernel((blocks,), (threads_per_block,), (
            packed_gpu, X_sorted,
            np.int64(self.n_samples), np.int64(n_variants),
            np.int64(bpv)
        ))
        if stream:
            stream.synchronize()
        else:
            cp.cuda.Device().synchronize()

        inverse_order = np.argsort(sort_order)
        X_gpu = X_sorted[:, cp.asarray(inverse_order)]
        del packed_gpu, X_sorted
        cp.get_default_memory_pool().free_all_blocks()
        return X_gpu

    def read_to_numpy(self, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        """Unpack to CPU numpy. Slow — use read_to_gpu where possible."""
        if end is None:
            end = self.n_variants
        n_variants = end - start
        packed = self.read_packed_bytes(start, end)
        packed_arr = np.frombuffer(packed, dtype=np.uint8).reshape(n_variants, -1)
        X = np.empty((self.n_samples, n_variants), dtype=np.float32)
        for v in range(n_variants):
            for s in range(self.n_samples):
                byte_idx = s // 4
                bit_shift = 6 - 2 * (s % 4)
                geno = (packed_arr[v, byte_idx] >> bit_shift) & 0x03
                X[s, v] = float(geno) if geno < 3 else np.nan
        return X

    def read_to_gpu(self, start: int = 0, end: Optional[int] = None,
                    stream: Optional['cp.cuda.Stream'] = None) -> 'cp.ndarray':
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")
        if end is None:
            end = self.n_variants
        n_variants = end - start
        cp.cuda.Device(self.device).use()

        packed = self.read_packed_bytes(start, end)
        packed_arr = np.frombuffer(packed, dtype=np.uint8)
        packed_gpu = cp.asarray(packed_arr)
        X_gpu = cp.empty((self.n_samples, n_variants), dtype=cp.float32)
        kernel = _get_unpack_kernel()
        threads_per_block = 256
        blocks = (self.n_samples + threads_per_block - 1) // threads_per_block
        kernel((blocks,), (threads_per_block,), (
            packed_gpu, X_gpu,
            np.int64(self.n_samples), np.int64(n_variants),
            np.int64(self.bytes_per_variant)
        ))
        if stream:
            stream.synchronize()
        else:
            cp.cuda.Device().synchronize()
        return X_gpu

    def fused_univariate(self, y_centered: Union[np.ndarray, 'cp.ndarray'],
                         start: int = 0, end: Optional[int] = None,
                         lambda_reg: float = 0.0,
                         stream=None) -> Tuple['cp.ndarray', 'cp.ndarray', 'cp.ndarray']:
        """Fused univariate regression directly from packed genotypes (no float32 matrix)."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available")
        if end is None:
            end = self.n_variants
        n_variants = end - start
        cp.cuda.Device(self.device).use()

        if isinstance(y_centered, np.ndarray):
            y_gpu = cp.asarray(y_centered.astype(np.float32))
        else:
            y_gpu = y_centered

        mu_x = cp.asarray(self.mu_x[start:end])
        sxx = cp.asarray(self.sxx[start:end])

        packed = self.read_packed_bytes(start, end)
        packed_gpu = cp.asarray(np.frombuffer(packed, dtype=np.uint8))
        num_gpu = cp.empty(n_variants, dtype=cp.float32)

        kernel = _get_fused_univariate_kernel()
        threads_per_block = 256
        blocks = (n_variants + threads_per_block - 1) // threads_per_block
        kernel((blocks,), (threads_per_block,), (
            packed_gpu, y_gpu, mu_x, sxx, num_gpu,
            np.int64(self.n_samples), np.int64(n_variants),
            np.int64(self.bytes_per_variant), np.float32(lambda_reg)
        ), stream=stream)
        if stream:
            stream.synchronize()
        else:
            cp.cuda.Device().synchronize()

        if stream:
            with stream:
                den_gpu = sxx + lambda_reg
                den_gpu = cp.maximum(den_gpu, 1e-20)
                beta_gpu = num_gpu / den_gpu
        else:
            den_gpu = sxx + lambda_reg
            den_gpu = cp.maximum(den_gpu, 1e-20)
            beta_gpu = num_gpu / den_gpu
        return beta_gpu, num_gpu, den_gpu

    def info(self) -> dict:
        return {
            'path': self.path,
            'version': self.version,
            'encoding': ['2bit', 'uint8', 'float16', 'float32'][self.encoding],
            'n_samples': self.n_samples,
            'n_variants': self.n_variants,
            'bytes_per_variant': self.bytes_per_variant,
            'has_missing': self.has_missing,
            'has_gidx_map': self.has_gidx_map,
            'file_size_gb': os.path.getsize(self.path) / 1e9,
            'data_size_gb': (self.n_variants * self.bytes_per_variant) / 1e9,
        }


class CugenReaderPinned(CugenReader):
    """Pinned-host ring buffer + async-H2D variant of CugenReader.

    1.6-1.85× faster sequential read on the cugen genotype block (per
    session-6 benchmarks). Header / stats / random-access paths inherit
    from CugenReader unchanged; only read_to_gpu is overridden.

    Env-var overrides:
        CUGEN_CHUNK_MB   — pinned buffer size in MB (default 64)
        CUGEN_N_BUFFERS  — ring depth (default 2)
    """

    def __init__(self, path: str, device: int = 0, chunk_mb: int = 64,
                 use_odirect: bool = False, n_buffers: int = 2):
        super().__init__(path, device=device, use_gds=False)
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available — pinned reader requires GPU")

        env_chunk = os.environ.get('CUGEN_CHUNK_MB')
        if env_chunk:
            try:
                chunk_mb = int(env_chunk)
            except ValueError:
                pass
        env_buffers = os.environ.get('CUGEN_N_BUFFERS')
        if env_buffers:
            try:
                n_buffers = int(env_buffers)
            except ValueError:
                pass
        self.chunk_mb = chunk_mb
        self.use_odirect = use_odirect
        self.n_buffers = max(1, n_buffers)

        flags = os.O_RDONLY
        if use_odirect:
            flags |= os.O_DIRECT
        self._raw_fd = os.open(path, flags)

        chunk_bytes = chunk_mb * 1024 * 1024
        if use_odirect:
            chunk_bytes = ((chunk_bytes + 4095) // 4096) * 4096
        self._chunk_bytes = chunk_bytes

        self._pinned = []
        self._pinned_np = []
        for _ in range(self.n_buffers):
            mem = cp.cuda.alloc_pinned_memory(chunk_bytes)
            self._pinned.append(mem)
            self._pinned_np.append(np.frombuffer(mem, dtype=np.uint8, count=chunk_bytes))

        self._h2d_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(self.n_buffers)]

    def close(self):
        super().close()
        if hasattr(self, '_raw_fd') and self._raw_fd is not None:
            try:
                os.close(self._raw_fd)
            except OSError:
                pass
            self._raw_fd = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read_to_gpu(self, start: int = 0, end: Optional[int] = None,
                    stream: Optional['cp.cuda.Stream'] = None) -> 'cp.ndarray':
        if end is None:
            end = self.n_variants
        n_variants = end - start
        if n_variants == 0:
            return cp.empty((self.n_samples, 0), dtype=cp.float32)

        bpv = self.bytes_per_variant
        total_bytes = n_variants * bpv
        base_offset = self.data_offset + start * bpv
        cp.cuda.Device(self.device).use()

        packed_gpu = cp.empty(total_bytes, dtype=cp.uint8)
        X_gpu = cp.empty((self.n_samples, n_variants), dtype=cp.float32)

        chunk = self._chunk_bytes
        n_chunks = (total_bytes + chunk - 1) // chunk

        for i in range(n_chunks):
            buf_idx = i % self.n_buffers
            stream_h2d = self._h2d_streams[buf_idx]
            buf_np = self._pinned_np[buf_idx]
            offset_in = i * chunk
            n_to_read = min(chunk, total_bytes - offset_in)

            if i >= self.n_buffers:
                stream_h2d.synchronize()

            view = memoryview(buf_np)[:n_to_read]
            n_read = os.preadv(self._raw_fd, [view], base_offset + offset_in)
            if n_read != n_to_read:
                raise IOError(
                    f"preadv short read: got {n_read}, expected {n_to_read} "
                    f"at offset {base_offset + offset_in}"
                )

            cp.cuda.runtime.memcpyAsync(
                packed_gpu.data.ptr + offset_in,
                buf_np.ctypes.data,
                n_to_read,
                cp.cuda.runtime.memcpyHostToDevice,
                stream_h2d.ptr,
            )

        for s in self._h2d_streams:
            s.synchronize()

        kernel = _get_unpack_kernel()
        threads_per_block = 256
        blocks = (self.n_samples + threads_per_block - 1) // threads_per_block
        kernel((blocks,), (threads_per_block,), (
            packed_gpu, X_gpu,
            np.int64(self.n_samples), np.int64(n_variants),
            np.int64(self.bytes_per_variant),
        ))
        if stream is not None:
            stream.synchronize()
        else:
            cp.cuda.Device().synchronize()
        del packed_gpu
        return X_gpu


def read_cugen_header(path: str) -> dict:
    """Parse a .cugen header without opening for I/O. Returns metadata dict.

    Lightweight — does NOT load stats or mmap the data block. Use for `info`
    subcommand and config validation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'rb') as f:
        header = f.read(HEADER_SIZE)
    if header[0:8] != CUGEN_MAGIC:
        raise ValueError(f"Invalid magic bytes: {header[0:8]!r}")
    return {
        'path': path,
        'version': struct.unpack_from('<I', header, 8)[0],
        'encoding': ['2bit', 'uint8', 'float16', 'float32'][
            struct.unpack_from('<I', header, 12)[0]
        ],
        'n_samples': struct.unpack_from('<Q', header, 16)[0],
        'n_variants': struct.unpack_from('<Q', header, 24)[0],
        'bytes_per_variant': struct.unpack_from('<Q', header, 32)[0],
        'file_size_gb': os.path.getsize(path) / 1e9,
    }


def read_cugen(path: str, device: int = 0,
               use_pinned: Optional[bool] = None) -> CugenReader:
    """Open a .cugen file. Returns a CugenReader (or CugenReaderPinned).

    use_pinned: None (default) → check env USE_PINNED_READER=1; True/False
    force the choice. Pinned reader requires CuPy.
    """
    if use_pinned is None:
        use_pinned = os.environ.get('USE_PINNED_READER', '0') == '1'
    if use_pinned and HAS_CUPY:
        return CugenReaderPinned(path, device=device)
    return CugenReader(path, device=device)


def write_cugen(*a, **kw):
    """Write a cugen file. v0.2 — for v0.1 use the existing pgen_to_cugen pipeline."""
    raise NotImplementedError(
        "cugen.io.write_cugen is planned for v0.2 — see README roadmap. "
        "For v0.1, build cugen files using the legacy build_*_cugen.sh scripts."
    )
