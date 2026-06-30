# Performance & Memory Improvement Plan

## Guiding philosophy

Profile first, fix what the profile says, prefer *local* patches over architectural rewrites. The package's hot loop is `generate_counterfactual` → `update!` → `generate_perturbations` → `grad_search_opt` → `Optimisers.update`, repeated up to `max_iter` (default 100) times per counterfactual. Every allocation in that loop is multiplied by ~100, so even small per-iteration wins compound.

## Libraries (dev/test-only)

| Library | Purpose |
|---|---|
| `BenchmarkTools.jl` | Micro-benchmarks (`@btime`, `@benchmark`). Already in `test/Project.toml`. |
| `Profile.jl` (stdlib) | Stack-sampling profiler. `@profile` + `Profile.print`. |
| `ProfileView.jl` | Flame-graph UI. Add to `benchmark/Project.toml`. |
| `ProfileCanvas.jl` | Alternative flame-graph renderer (HTML/VS Code). Optional. |
| `JET.jl` | Static dispatch finding. `@report_opt` on hot functions. |
| `Cthulhu.jl` | Interactive `@descend`/`@descend_code_warntype`. |
| `TimerOutputs.jl` | Optional section-level timing. |
| `pprof` | Heap-profile visualisation via `--heap-profile-hp`. |

Built-in tools: `@code_warntype`, `--track-allocation=user`, `--heap-profile-hp`.

## Phase 0 — Reproducible benchmark harness

1. Create `benchmark/Project.toml` with `BenchmarkTools`, `ProfileView`, `Cthulhu`, `TimerOutputs`, `JET`, plus package via `dev` path.
2. Write `benchmark/benchmarks.jl` with a `BenchmarkTools.SUITE` covering:
   - **End-to-end**: `@btime generate_counterfactual(...)` on linearly-separable toy data and a small MLP.
   - **Per-iteration**: `update!(ce)`, `generate_perturbations(generator, ce)`, `grad_search_opt(generator, ce)`, `grad_loss(generator, ce)`, `decode_state(ce)`, `apply_mutability(ce, grad)`.
   - **Memory**: `@benchmark ... memory=true`.
3. Run once, save `BenchmarkTools.save("benchmark/baseline.json")`.

## Phase 1 — Profile the hot loop

1. `@report_opt` (JET) on `update!`, `generate_perturbations`, `grad_search_opt`, `grad_loss`, `grad_pen`, `decode_state`, `apply_mutability`.
2. `@code_warntype` on the same functions.
3. `@profile` end-to-end, `ProfileView.view()`.
4. `--track-allocation=user` + `Profile.clear_malloc_data()` for per-line allocation counts.
5. `--heap-profile-hp=` if memory growth is the concern.

## Phase 2 — Low-hanging fruit (local patches, no refactors)

### 2a. Kill O(n²) path growth
`src/counterfactuals/search.jl:22` — replace `[path..., new]` with `push!(path, new)`. Also `sizehint!` at init.

### 2b. Remove gratuitous `deepcopy`
`src/generators/generate_perturbations.jl:11` and `gradient_based/generate_perturbations.jl:9` — replace `deepcopy` with `copy` or remove entirely (prove correctness first).

### 2c. Eliminate per-iteration `convert.` broadcast
`generate_perturbations.jl:15` and `gradient_based/generate_perturbations.jl:13` — guard with type check or delete if types match upstream.

### 2d. Replace allocating `_replace_nans`
`src/generators/gradient_based/utils.jl:7` — use `replace!` or fuse NaN check into `apply_mutability`.

### 2e. Allocation-free `get_target_index`
`src/global_utils.jl:143` — `findfirst(==(target), y_levels)` instead of `findall(...)[1]`.

### 2f. Clean up `distance_mad` micro-allocations
`src/objectives/penalties.jl:22-33` — replace `mad = []; push!(...)` with direct use; replace `∈ collect(keys(...))` with `haskey`.

### 2g. Avoid double `apply_mutability` per iteration
Cache constrained gradient or pass it to `conditions_satisfied`.

### 2h. `find_potential_neighbours` allocation cleanup
`src/counterfactuals/utils.jl:88` — replace `collect` + `map` + `reduce(hcat, ...)` with single preallocated matrix.

### 2i. Cache `target_idx`
Compute once in `initialize!`, store on search state, read back instead of `get_target_index` each iteration.

## Phase 3 — Concrete `SearchState` struct (medium effort, high payoff)

Replace `ce.search::Union{Dict,Nothing}` with a concrete `mutable struct SearchState`. This eliminates the `Any`-typed dict lookups that are the single largest source of type instability. Fields: `iteration_count`, `mutability`, `path`, `times_changed_features`, `opt_state`, `converged`, `target_idx`, `mad_features`, `potential_neighbours`, `ad_backend`, `prep_loss`, `prep_pen`.

This is a contained refactor: only `initialize!`, `update!`, `termination.jl`, `path_tracking.jl`, `loss.jl`, `penalties.jl` need updating.

## Phase 4 — Parametric `CounterfactualExplanation` (medium effort)

Make `CounterfactualExplanation` parametric on its array type fields (`factual`, `counterfactual_state`, `counterfactual`). Lets the compiler specialise on `Matrix{Float32}` vs `Matrix{Float64}`. Mechanical change to constructors, `getproperty`/`setproperty!`, and `FlattenedCE`.

## Phase 5 — Verify

1. Re-run benchmark suite vs `baseline.json` (`BenchmarkTools.compare`).
2. Re-run `@report_opt` — dispatch reports should be gone.
3. `Pkg.test` — confirm no behavioural regressions.
4. CHANGELOG entry under "Performance".