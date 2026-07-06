# Host Format Migration Decision: clsvof_ls_nn_static_bubble

- run_id: `host_format_migration_clsvof_ls_nn_static_bubble_20260630T140135192844Z`
- status: `BLOCKED`
- recommendation: `do_not_migrate_in_this_plan`
- source_case: `cfd_applications_cleanroom/cases/stationary/stationary_clsvof_nn.c`
- accepted_metric: `Ca_tail_max`

| gate | pass |
|---|---|
| `HF-G0` | `false` |
| `HF-G1` | `false` |
| `HF-G2` | `false` |
| `HF-G3` | `false` |
| `HF-G4` | `false` |

Interpretation:

- `PASS` means this application can be presented as a VOF-HF host-format migration positive control.
- `BLOCKED` means the application must not be described as solved by VOF-HF under this plan.
- No neural curvature injection is deployable in the VOF-HF host in this plan.
