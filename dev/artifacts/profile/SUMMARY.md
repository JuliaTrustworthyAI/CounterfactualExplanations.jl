# Profiling Summary — Phase 1

**Date:** 2026-08-09  
**Environment:** `--project=test`, Julia 1.12.6, JET.jl v0.9, CounterfactualExplanations v1.5.0 (local dev)  
**Setup:** Linear model, `GenericGenerator`, linearly-separable toy data (2 features, 1 counterfactual)

## Overview

| Function | JET issues | `@code_warntype` return type | Key problem |
|---|---|---|---|
| `update!` | 646 | `Any` | `ce.search::Union{Nothing,Dict}` → all dict lookups return `Any` |
| `generate_perturbations` | (large) | `Any` | `deepcopy` + `convert.` broadcast + `propose_state` returns `Any` |
| `propose_state` | (large) | `Any` | `ce.search[:opt_state]` returns `Any` |
| `grad_search_opt` | (large) | `Any` | `grad_loss` and `grad_pen` both return `Any` |
| `grad_loss` | (large) | `Any` | `get!(ce.search, ...)` returns `Any` for `backend`, `prep` |
| `grad_pen` | (large) | `Any` | Same as `grad_loss` |
| `decode_state` | (large) | `Matrix` (concrete!) | Minor issues only |
| `apply_mutability` | 12 | `Matrix{Float32}` (concrete!) | `ce.search[:mutability]` returns `Any` |

## Root causes (ordered by impact)

### 1. `ce.search::Union{Nothing,Dict}` — the dominant type instability

Every `ce.search[:foo]` returns `Any`, propagating through the entire hot loop.

**Evidence** (`codewarntype_update.txt`):
```
%27 = Base.getproperty(ce, :search)::UNION{NOTHING, DICT}
%28 = Base.getindex(%27, :times_changed_features)::ANY
%40 = Base.getindex(%39, :iteration_count)::ANY
%45 = Base.getindex(%44, :path)::ANY
```

**Affected files:**
- `src/counterfactuals/search.jl:18,20,21,22,43` — `update!`, `apply_mutability`
- `src/counterfactuals/termination.jl:16,25` — `steps_exhausted`, `total_steps`
- `src/counterfactuals/path_tracking.jl:7` — `path`
- `src/counterfactuals/printing.jl:2` — `show`
- `src/counterfactuals/generate_counterfactual.jl:163` — `converged`
- `src/convergence/Convergence.jl:21` — `converged`
- `src/convergence/max_iter.jl:29` — `converged`
- `src/generators/gradient_based/generate_perturbations.jl:40` — `opt_state`
- `src/generators/gradient_based/loss.jl:25,29,93,97` — `ad_backend`, `prep_loss`, `prep_pen`
- `src/objectives/penalties.jl:24,30,140,309` — `mad_features`, `potential_neighbours`
- `src/evaluation/plausibility/plausibility.jl:25,31,67,73` — `potential_neighbours`
- `src/evaluation/faithfulness/utils.jl:303,311` — `energy_sampler`

**Fix:** Phase 3 — `SearchState` struct.

### 2. `ce.counterfactual_state::AbstractArray` and `ce.factual::AbstractArray`

The struct fields are typed as `AbstractArray`, not a concrete array type. This prevents the compiler from specializing on `Matrix{Float32}`.

**Evidence** (`codewarntype_update.txt`):
```
%9  = Base.getproperty(ce, :counterfactual_state)::ABSTRACTARRAY
%15 = Base.getproperty(ce, :counterfactual)        # decoded from AbstractArray
```

**Evidence** (`codewarntype_generate_perturbations.txt`):
```
counterfactual_state::ABSTRACTARRAY
new_counterfactual_state::ANY
grad_ce_state::ANY
Body::ANY
```

**Fix:** Phase 4 — parametric `CounterfactualExplanation`.

### 3. `deepcopy` in `generate_perturbations`

JET flags runtime dispatch in `deepcopy` of `AbstractArray`.

**Evidence** (`jet_update.txt` lines 4-16):
```
││┌ deepcopy(x::AbstractArray) @ Base ./deepcopy.jl:28
│││┌ isbitstype(t::Type{<:AbstractArray{T, N}} where {T, N}) @ Base ./runtime_internals.jl:836
││││ runtime dispatch detected: (%3::DataType).flags::Any
││└────────────────────
││┌ deepcopy(x::AbstractArray) @ Base ./deepcopy.jl:29
│││ runtime dispatch detected: Base.deepcopy_internal(x::AbstractArray, %6::IdDict{Any, Any})::Any
```

**Files:** `src/generators/generate_perturbations.jl:11`, `src/generators/gradient_based/generate_perturbations.jl:9`

**Fix:** Step 2b — replace `deepcopy` with `copy` or remove.

### 4. O(n²) path growth in `update!`

**Evidence** (`codewarntype_update.txt` line 58):
```
%48 = Core._apply_iterate(Base.iterate, Base.vect, %45, %47)::ANY
```

This is `[ce.search[:path]..., ce.counterfactual_state]` — allocates a new array every iteration, copying all previous elements.

**File:** `src/counterfactuals/search.jl:22`

**Fix:** Step 2a — `push!` instead of `[..., new]`.

### 5. `convert.` broadcast in `generate_perturbations`

**Evidence** (`codewarntype_generate_perturbations.txt` lines 24-30):
```
%14 = Base.getproperty(ce, :factual)::ABSTRACTARRAY
%16 = (%14)(%15)::ANY                    # eltype(ce.factual) is Any
%18 = Base.broadcasted(%13, %16, %17)::ANY
grad_ce_state = Base.materialize(%18)     # type unstable
```

**File:** `src/generators/generate_perturbations.jl:15`, `src/generators/gradient_based/generate_perturbations.jl:13`

**Fix:** Step 2c — remove or guard with type check.

### 6. `apply_mutability` — `mutability::Any` from dict lookup

**Evidence** (`codewarntype_apply_mutability.txt`):
```
%123 = Base.getproperty(ce, :search)::UNION{NOTHING, DICT}
mutability = Base.getindex(%123, :mutability)   # ::ANY
i::ANY, case::ANY, val::ANY                      # all downstream
```

Despite the `@inbounds` loop and `similar()` output being `Matrix{Float32}`, the loop body is fully type-unstable because `mutability` comes from a dict lookup.

**File:** `src/counterfactuals/search.jl:43`

**Fix:** Phase 3 — `SearchState.mutability::Vector{Symbol}`.

## Key findings (top 5, prioritized by hot-loop impact)

1. **`ce.search::Union{Dict,Nothing}`** — single largest source of type instability. Every hot-loop function reads from this dict. Affects ~20 call sites across 12 files. **Phase 3** will fix this comprehensively.

2. **`ce.counterfactual_state::AbstractArray`** — prevents specialization on `Matrix{Float32}`. Makes `generate_perturbations` return `Any`, propagating to `update!`. **Phase 4** will fix this.

3. **`deepcopy` of `AbstractArray`** — runtime dispatch in `deepcopy.jl`. **Step 2b** will fix this trivially.

4. **O(n²) path growth** — `[path..., new]` copies the entire path every iteration. **Step 2a** will fix this trivially.

5. **`convert.` broadcast** — allocates per-iteration with type-unstable eltype. **Step 2c** will fix this.

## Notes

- JET `@report_opt` produces very verbose output (646+ issues for `update!` alone) because type-unstable code causes JET to recursively flag every downstream runtime dispatch in Base Julia internals (e.g., `show.jl`, `promotion.jl`). The vast majority of these will disappear once the root causes above are fixed.
- `decode_state` and `apply_mutability` have concrete return types (`Matrix` / `Matrix{Float32}`) — they're the "best" functions in the hot loop, but still have internal type instabilities from dict lookups.
- `apply_mutability` has a large `@code_warntype` output (247 lines) because the `@warn` macro expands to extensive logging code. This is cosmetic, not a performance issue.
