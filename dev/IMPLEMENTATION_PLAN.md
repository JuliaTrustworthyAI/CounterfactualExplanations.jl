# Step-by-Step Implementation Plan

Derived from [`PLAN.md`](./PLAN.md). Each step is scoped to be self-contained and executable by a less capable agent. After every step: run `Pkg.test()`, compare against `benchmark/baseline.json`, hand back to user for review/commit.

## Status Legend

- [ ] pending · ▶️ in progress · ✅ done

---

## Phase 1 — Profile the hot loop

### Step 1.1 — Run JET `@report_opt` and `@code_warntype` on hot functions

- [ ] **1.1** Save profiling artefacts under `dev/artifacts/profile/`
- Create `dev/profile_jet.jl` that loads `CounterfactualExplanations` + a linear `ce` (reuse setup from `benchmark/benchmarks.jl`) and runs `JET.@report_opt` on: `update!`, `generate_perturbations` (both methods), `propose_state`, `grad_search_opt`, `grad_loss`, `grad_pen`, `decode_state`, `apply_mutability`.
- Run `@code_warntype` on the same and dump to text files.
- Save outputs to `dev/artifacts/profile/jet.txt` and `dev/artifacts/profile/codewarntype_<fn>.txt`.
- **Deliverable**: artefacts checked in, plus a short `dev/artifacts/profile/SUMMARY.md` listing every dispatch/instability found with `file:line` pointers.
- **No source changes.**

---

## Phase 2 — Low-hanging fruit (local patches)

Each step touches one or two files and is independently revertable.

### Step 2a — Kill O(n²) path growth

- [ ] **2a** `src/counterfactuals/search.jl:22` + `src/counterfactuals/core_struct.jl:139`
- In `initialize!`: replace `ce.search[:path] = [ce.s′]` with `ce.search[:path] = sizehint!([ce.s′], ce.convergence.max_iter + 1)`.
- In `update!`: replace `ce.search[:path] = [ce.search[:path]..., ce.counterfactual_state]` with `push!(ce.search[:path], ce.counterfactual_state)`.
- Run `Pkg.test()`. Confirm `test/counterfactuals/path_tracking.jl` still passes.

### Step 2b — Remove gratuitous `deepcopy`

- [ ] **2b** `src/generators/generate_perturbations.jl:11` + `src/generators/gradient_based/generate_perturbations.jl:9`
- Replace `deepcopy(ce.counterfactual_state)` with `copy(ce.counterfactual_state)` in both `generate_perturbations` methods. The value is only read, never mutated downstream (`propose_state` allocates a new array via `Optimisers.update`).
- Run `Pkg.test()`.

### Step 2c — Eliminate per-iteration `convert.` broadcast

- [ ] **2c** `src/generators/generate_perturbations.jl:15` + `src/generators/gradient_based/generate_perturbations.jl:13`
- Drop the `grad_ce_state = convert.(eltype(ce.factual), grad_ce_state)` line. Verify with a type assertion instead: if `eltype(grad_ce_state) !== eltype(ce.factual)` at init time, fix it once in `initialize!`; otherwise the broadcast is a no-op.
- Add a regression in `test/other/performance.jl` tightening `expected_allocs` downward (keep old value as a comment for the next step).
- Run `Pkg.test()`.

### Step 2d — Replace allocating `_replace_nans`

- [ ] **2d** `src/generators/gradient_based/utils.jl:6`
- Change `_replace_nans` to an in-place version operating on the gradient array before it is returned. Either:
  - rename to `_replace_nans!` and call `replace!(grad_ce_state, old_new)` on the already-allocated array inside `generate_perturbations` (both methods), or
  - fuse the NaN→0 replacement into `apply_mutability` (one extra branch per element, no extra pass).
- Prefer the `replace!` approach for minimal change.
- Update both call sites in `generate_perturbations.jl` files.
- Run `Pkg.test()`.

### Step 2e — Allocation-free `get_target_index`

- [ ] **2e** `src/global_utils.jl:141`
- Replace `findall(y_levels .== target)[1]` with `findfirst(==(target), y_levels)`.
- Keep the `@assert` guard.
- Run `Pkg.test()`.

### Step 2f — Clean up `distance_mad` micro-allocations

- [ ] **2f** `src/objectives/penalties.jl:14-34`
- Replace `mad = []; push!(mad, _mad)` with direct use of `_mad`.
- Replace `if !(:mad_features ∈ collect(keys(_dict)))` with `if !haskey(_dict, :mad_features)`.
- Run `Pkg.test()` (covers `test/other/objectives.jl`).

### Step 2g — Avoid double `apply_mutability` per iteration

- [ ] **2g** `src/generators/gradient_based/utils.jl:16` (`conditions_satisfied`) + `src/counterfactuals/search.jl`
- `conditions_satisfied` calls `grad_search_opt` + `apply_mutability` again, duplicating work already done in `update!`.
- Approach: cache the constrained gradient from `update!` on `ce.search[:last_constrained_grad]` and have `conditions_satisfied` read it. If missing (cold path), fall back to computing it.
- Touch only `search.jl::update!` (add cache write) and `utils.jl::conditions_satisfied` (read cache).
- Run `Pkg.test()`.

### Step 2h — `find_potential_neighbours` allocation cleanup

- [ ] **2h** `src/counterfactuals/utils.jl:81-90`
- Replace `reduce(hcat, map(x -> x[1], collect(candidates)))` with a single preallocated matrix:
  - `n_candidates` is known, feature dim is `size(data.X, 1)`.
  - Allocate `Matrix{eltype(data.X)}(undef, d, n_candidates)` and fill in a loop, or use `reduce(hcat, (x[1] for x in candidates))` (generator, no `collect`).
- Run `Pkg.test()`.

### Step 2i — Cache `target_idx`

- [ ] **2i** `src/counterfactuals/core_struct.jl::initialize!` + `src/objectives/penalties.jl::energy_constraint` + `src/counterfactuals/info_extraction.jl::target_probs`
- In `initialize!`: `ce.search[:target_idx] = get_target_index(ce.data.y_levels, ce.target)`.
- Replace `get_target_index(ce.data.y_levels, ce.target)` call sites in the hot path (`energy_constraint`, `target_probs`, `EnergyDifferential`) with `ce.search[:target_idx]`.
- Keep `get_target_index` public for external callers.
- Run `Pkg.test()`.

---

## Phase 3 — Concrete `SearchState` struct

### Step 3.1 — Introduce `SearchState` struct (additive, no behaviour change)

- [ ] **3.1** New file `src/counterfactuals/search_state.jl`
- Define `mutable struct SearchState` with fields:
  ```
  iteration_count::Int
  mutability::Vector{Symbol}
  path::Vector{<:AbstractArray}
  times_changed_features::AbstractArray
  opt_state
  converged::Bool
  target_idx::Int
  mad_features::Union{Nothing,AbstractArray}
  potential_neighbours::Union{Nothing,AbstractArray}
  ad_backend
  prep_loss
  prep_pen
  extra::Dict{Symbol,Any}   # escape hatch for callbacks, energy_sampler, etc.
  ```
- Add `Base.getindex(state::SearchState, k::Symbol)` and `Base.setindex!(state::SearchState, v, k::Symbol)` that dispatch to the concrete fields when known, else fall back to `extra`.
- `include("search_state.jl")` in `Counterfactuals.jl` before `core_struct.jl`.
- Keep `ce.search::Union{Dict,Nothing}` for now — do not switch the field type yet.
- Run `Pkg.test()` — no functional change yet.

### Step 3.2 — Switch `CounterfactualExplanation.search` to `SearchState`

- [ ] **3.2** `src/counterfactuals/core_struct.jl` + `src/counterfactuals/flatten.jl`
- Change field type `search::Union{Dict,Nothing}` → `search::Union{SearchState,Nothing}`.
- Update `initialize!` to construct a `SearchState(...)` instead of a `Dict(...)`.
- Update `FlattenedCE.search::Dict` → `search::SearchState` (or keep `Dict` by converting in `flatten`; pick whichever is less invasive — prefer keeping `FlattenedCE` on `SearchState`).
- The `getindex`/`setindex!` shim from 3.1 keeps every existing `ce.search[:foo]` site working unchanged.
- Update the two test sites that use `ce.search[:greeting]` and `delete!(ce.search, :energy_sampler)` to confirm they route through `extra`.
- Run `Pkg.test()`.

### Step 3.3 — Replace dict-pattern accessors with typed accessors

- [ ] **3.3** All `ce.search[:foo]` call sites
- Replace with `ce.search.foo` (direct field access) in:
  - `search.jl`: `iteration_count`, `times_changed_features`, `path`, `mutability`
  - `termination.jl`: `iteration_count`
  - `path_tracking.jl`: `path`
  - `printing.jl`: `iteration_count`
  - `generate_counterfactual.jl`: `converged`
  - `convergence/Convergence.jl`: `converged`
  - `convergence/max_iter.jl`: `iteration_count`
  - `generators/gradient_based/generate_perturbations.jl`: `opt_state`
  - `generators/gradient_based/loss.jl`: `ad_backend`, `prep_loss`, `prep_pen`
  - `objectives/penalties.jl`: `mad_features`, `potential_neighbours`
  - `evaluation/plausibility/plausibility.jl`: `potential_neighbours`
  - `evaluation/faithfulness/utils.jl`: `energy_sampler` (goes to `extra`)
  - `counterfactuals/utils.jl::adjust_shape!`: `mutability`
- Leave the `getindex`/`setindex!` shim in place as a fallback.
- Run `Pkg.test()`.

### Step 3.4 — Specialise `getproperty`/`setproperty!` for `SearchState`

- [ ] **3.4** `src/counterfactuals/search_state.jl`
- Define `Base.getproperty(::SearchState, sym::Symbol)` and `Base.setproperty!(::SearchState, sym::Symbol, val)` mirroring the alias pattern used in `core_struct.jl` if needed (likely not needed — keep simple).
- Ensure `hasfield`/`fieldnames` still work (don't shadow unless necessary).
- Run `Pkg.test()` + re-run JET on hot functions to confirm dispatch reports reduced.

---

## Phase 4 — Parametric `CounterfactualExplanation`

### Step 4.1 — Make `CounterfactualExplanation` parametric on array type

- [ ] **4.1** `src/counterfactuals/core_struct.jl`
- Change struct to:
  ```
  mutable struct CounterfactualExplanation{A<:AbstractArray, ...} <: AbstractCounterfactualExplanation
      factual::A
      target::RawTargetType
      target_encoded::EncodedTargetType
      counterfactual_state::A
      counterfactual::A
      ...
  end
  ```
- Keep `factual`, `counterfactual_state`, `counterfactual` sharing one type param `A`.
- Update inner constructor to infer `A` from `factual`.
- Update `getproperty`/`setproperty!` overrides to preserve type params.
- Run `Pkg.test()`.

### Step 4.2 — Update `FlattenedCE` and constructors

- [ ] **4.2** `src/counterfactuals/flatten.jl` + `core_struct.jl` outer constructor
- Make `FlattenedCE` parametric to match.
- Ensure `unflatten` reconstructs the right parametric type.
- Run `Pkg.test()`.

### Step 4.3 — Verify parametric specialisation pays off

- [ ] **4.3** No code changes — verification only
- Run `@code_warntype` on `update!` and `generate_perturbations`; confirm array field accesses are now concrete.
- Re-run JET `@report_opt`; record before/after in `dev/artifacts/profile/SUMMARY.md`.

---

## Phase 5 — Verify

### Step 5.1 — Re-run benchmarks vs baseline

- [ ] **5.1**
- `julia --project=benchmark benchmark/run_benchmarks.jl` → produces new `benchmark/results.json`.
- Write `dev/compare_benchmarks.jl` that loads `baseline.json` + `results.json` and prints `BenchmarkTools.compare`.
- Save summary to `dev/artifacts/profile/BENCHMARK_BEFORE_AFTER.md`.

### Step 5.2 — Re-run JET and confirm dispatch reports gone

- [ ] **5.2**
- Re-run `dev/profile_jet.jl`.
- Diff against Phase 1 artefacts. Record in `SUMMARY.md`.

### Step 5.3 — Full test suite + performance regression guard

- [ ] **5.3**
- `Pkg.test()` green.
- Tighten `expected_allocs` in `test/other/performance.jl` to the new measured value (keep a comment with the old bound).
- Add a `@test_broken` or comment referencing the target so future regressions are caught.

### Step 5.4 — CHANGELOG entry

- [ ] **5.4** `CHANGELOG.md`
- Add entry under "Performance" summarising the wins (cite benchmark numbers from 5.1).

---

## Notes for implementers

- Every step ends with `Pkg.test()`. If tests fail, fix the code, not the tests.
- Do not commit. The user commits after reviewing each step.
- If a step reveals an unforeseen dependency, stop and report back before expanding scope.
- Keep changes minimal: one step = one concern = small diff.
