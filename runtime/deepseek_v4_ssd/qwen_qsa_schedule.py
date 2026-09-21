"""Host-only planning for QSA attention chunks; estimates are not process limits."""
from __future__ import annotations

QSA_WORKSPACE_BYTES = 256 * 1024**2


def query_chunk_size(requested: int, *, selected_rows: int, kv_heads: int,
                     query_heads: int, head_dim: int, element_bytes: int,
                     index_blocks: int, index_heads: int) -> int:
    """Bound one attention chunk's estimated gather, score and index workspace.

    This excludes existing model/KV storage, allocator caching, output tensors,
    graph lifetimes and backend-private workspace. It is not a hard RAM cap.
    """
    if type(requested) is not int or not 1 <= requested <= 128:
        raise ValueError("QSA query chunk must be an integer from 1 through 128")
    dimensions = (selected_rows, kv_heads, query_heads, head_dim, element_bytes,
                  index_blocks, index_heads)
    if any(type(x) is not int or x < 0 for x in dimensions):
        raise ValueError("QSA workspace dimensions must be nonnegative integers")
    per_query = (2 * selected_rows * kv_heads * head_dim * element_bytes
                 + 12 * selected_rows * query_heads
                 + 4 * index_blocks * (index_heads + 3))
    return max(1, min(requested, QSA_WORKSPACE_BYTES // max(1, per_query)))
