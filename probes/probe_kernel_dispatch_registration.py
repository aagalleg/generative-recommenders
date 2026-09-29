"""Confirm fbgemm-xpu registered an XPU dispatch key for the ops DLRM-v3 needs.

Checks registration only. A registered op can still fail at runtime, so the
TBE tests and the end-to-end run remain the real evidence.
"""

import torch
import fbgemm_gpu  # noqa: F401  registers the CPU/meta side and the schemas
import fbgemm_xpu  # noqa: F401  registers the XPU kernels

# Stage A carries the embedding-lookup critical path; the jagged ops arrive in
# Stage B (PR #96) and are expected to be missing here.
STAGE_A = [
    "dense_embedding_codegen_lookup_function",
    "split_embedding_nobag_backward_codegen_dense_unweighted_exact_xpu",
    "dense_embedding_nobag_forward_unweighted_xpu",
    "asynchronous_complete_cumsum",
    "asynchronous_exclusive_cumsum",
    "permute_1D_sparse_data",
    "permute_2D_sparse_data",
    "block_bucketize_sparse_features",
    "expand_into_jagged_permute",
    "jagged_index_select_2d_forward",
    "get_infos_metadata",
    "invert_permute",
]

STAGE_B = [
    "jagged_to_padded_dense",
    "dense_to_jagged",
    "jagged_dense_elementwise_add_jagged_output",
]


def report(title, ops):
    print(f"=== {title} ===")
    n_xpu = 0
    for op in ops:
        qualified = f"fbgemm::{op}"
        try:
            table = torch._C._dispatch_dump(qualified)
        except RuntimeError as exc:
            print(f"  {op:<62} NO SCHEMA ({exc.__class__.__name__})")
            continue
        keys = [
            line.split(":")[0].strip()
            for line in table.splitlines()
            if ":" in line and not line.startswith(" ")
        ]
        has_xpu = any(k in ("XPU", "AutogradXPU", "CompositeExplicitAutograd") for k in keys)
        # CompositeExplicitAutograd covers XPU implicitly, so call it out separately.
        explicit = "XPU" in keys
        n_xpu += explicit
        mark = "XPU" if explicit else ("composite" if has_xpu else "-- MISSING --")
        print(f"  {op:<62} {mark}")
    print(f"  -> {n_xpu}/{len(ops)} with an explicit XPU key\n")
    return n_xpu


a = report("Stage A (expected present)", STAGE_A)
b = report("Stage B / PR #96 (expected absent until Stage B)", STAGE_B)
print(f"SUMMARY stage_a_xpu={a}/{len(STAGE_A)} stage_b_xpu={b}/{len(STAGE_B)}")
