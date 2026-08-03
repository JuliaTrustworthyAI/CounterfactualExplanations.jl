# Performance & Memory Improvement Plan

**Approach:** Profile first, fix what the profile says, prefer local patches over architectural rewrites.

## Phase 0 — Benchmark harness (done)

- `benchmark/Project.toml` + `benchmark/benchmarks.jl` + `benchmark/run_benchmarks.jl`
- Covers e2e (Linear, MLP) and per-iteration hot functions
- Baseline saved to `benchmark/baseline.json`

## Phase 1 — Profile the hot loop

- JET `@report_opt` + `@code_warntype` on `update!`, `generate_perturbations`, `grad_search_opt`, `grad_loss`, `grad_pen`, `decode_state`, `apply_mutability`
- `@profile` end-to-end, `ProfileView.view()`
- `--track-allocation=user` for per-line allocation counts

## Phase 2 — Low-hanging fruit (local patches)

| # | Item | File |
|---|------|------|
| 2a | Kill O(n²) path growth | `src/counterfactuals/search.jl:22` |
| 2b | Remove gratuitous `deepcopy` | `src/generators/generate_perturbations.jl:11`, `gradient_based/generate_perturbations.jl:9` |
| 2c | Eliminate per-iteration `convert.` broadcast | `generate_perturbations.jl:15` and sibling |
| 2d | Replace allocating `_replace_nans` | `src/generators/gradient_based/utils.jl:7` |
| 2e | Allocation-free `get_target_index` | `src/global_utils.jl:143` |
| 2f | Clean up `distance_mad` micro-allocations | `src/objectives/penalties.jl:22-33` |
| 2g | Avoid double `apply_mutability` per iteration | Cache constrained gradient |
| 2h | `find_potential_neighbours` allocation cleanup | `src/counterfactuals/utils.jl:88` |
| 2i | Cache `target_idx` | Compute once in `initialize!` |

## Phase 3 — Concrete `SearchState` struct

Replace `ce.search::Union{Dict,Nothing}` with a `mutable struct SearchState` to eliminate `Any`-typed dict lookups.

## Phase 4 — Parametric `CounterfactualExplanation`

Make `CounterfactualExplanation` parametric on array types (`factual`, `counterfactual_state`, `counterfactual`).

## Phase 5 — Verify

1. Re-run benchmark suite vs `baseline.json` (`BenchmarkTools.compare`)
2. Re-run `@report_opt` — dispatch reports should be gone
3. `Pkg.test` — confirm no regressions
4. CHANGELOG entry under "Performance"