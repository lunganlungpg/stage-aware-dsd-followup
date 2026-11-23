# stageaware (Python)

Companion Python package for *Stage-aware follow-up designs for definitive screening experiments*.

```bash
pip install "git+https://github.com/lunganlungpg/stage-aware-dsd-followup#subdirectory=python"
```

| Function | Paper |
|---|---|
| `dsd(m)`, `paley_conference(q)` | Conference-matrix DSDs, m = 5..14 (Section 5.1) |
| `FollowUpProblem(X1)` | Fixes the projected Stage 1 design; exposes `r1`, `q`, `S1`, `Q`, `V`, `Omega` |
| `.psi(X2)`, `.phi_I(X2)`, `.contrast(X2)`, `.prediction_variance(X2, x)` | Proposition 1, eqs (4)–(6) |
| `.unblocked_I`, `.confounding_index`, `.stage_cost` | Proposition 2, eqs (7)–(8) |
| `.S_pool`, `.pooled_contrast`, `.leverage`, `.pooled_gap` | Proposition 3, eqs (9)–(10) |
| `.rank_adjusted`, `.residual_df`, `minimum_followup_runs`, `useful_run_counts` | Theorem 1 and Section 3 |
| `rank_two_update`, `updated_value`, `coordinate_exchange` | Proposition 4 and Section 4.1 |
| `trace_bound`, `relaxation_bound` | Proposition 5 and Theorem 2 |

Run `pytest tests` and `python examples/reproduce_tables.py`.
