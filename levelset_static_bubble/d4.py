import numpy as np

from levelset_static_bubble.features import PHI9_OFFSETS

D4_MATRICES = (
    np.asarray([[1, 0], [0, 1]], dtype=np.int64),
    np.asarray([[0, -1], [1, 0]], dtype=np.int64),
    np.asarray([[-1, 0], [0, -1]], dtype=np.int64),
    np.asarray([[0, 1], [-1, 0]], dtype=np.int64),
    np.asarray([[1, 0], [0, -1]], dtype=np.int64),
    np.asarray([[-1, 0], [0, 1]], dtype=np.int64),
    np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    np.asarray([[0, -1], [-1, 0]], dtype=np.int64),
)


def transform_raw27(raw: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float32)
    if raw_arr.shape != (27,):
        raise ValueError(f"raw27 vector must have shape (27,), got {raw_arr.shape}.")
    p9 = raw_arr[:9]
    nx = raw_arr[9:18]
    ny = raw_arr[18:27]
    lookup = {offset: k for k, offset in enumerate(PHI9_OFFSETS)}
    mat_i = np.asarray(matrix, dtype=np.int64)
    mat_f = np.asarray(matrix, dtype=np.float32)
    out_p = np.empty(9, dtype=np.float32)
    out_nx = np.empty(9, dtype=np.float32)
    out_ny = np.empty(9, dtype=np.float32)
    for k, offset in enumerate(PHI9_OFFSETS):
        src_offset = tuple(mat_i.T @ np.asarray(offset, dtype=np.int64))
        src = lookup[src_offset]
        vec = np.asarray([nx[src], ny[src]], dtype=np.float32)
        transformed_vec = mat_f @ vec
        out_p[k] = p9[src]
        out_nx[k] = transformed_vec[0]
        out_ny[k] = transformed_vec[1]
    return np.concatenate([out_p, out_nx, out_ny]).astype(np.float32, copy=False)
