import DifferentiationInterface as DI

"A base type for a style of process."
abstract type PenaltyRequirements end

"By default, penalties have no extra requirements."
struct NoPenaltyRequirements <: PenaltyRequirements end
PenaltyRequirements(::Type) = NoPenaltyRequirements()

"Penalties that need access to neighbors in the target class."
struct NeedsNeighbours <: PenaltyRequirements end

"The `distance_from_target` method needs neighbors in the target class."
PenaltyRequirements(::Type{<:typeof(distance_from_target)}) = NeedsNeighbours()

# Implementing trait behaviour:
needs_neighbours(x::T) where {T} = needs_neighbours(PenaltyRequirements(T), x)
needs_neighbours(::NoPenaltyRequirements, x) = false
needs_neighbours(::NeedsNeighbours, x) = true

"""
    needs_neighbours(gen::AbstractGenerator)

Check if a generator needs access to neighbors in the target class.
"""
function needs_neighbours(gen::AbstractGenerator)
    # No penalty defined:
    if !hasfield(typeof(gen), :penalty)
        return false
    end
    # No penalty supplied:
    if isnothing(gen.penalty)
        return false
    end
    # All other cases:
    penalty = CounterfactualExplanations.flatten_penalty(gen.penalty)
    return any(needs_neighbours.(penalty))
end

"""
    needs_neighbours(ce::AbstractCounterfactualExplanation)

Check if a counterfactual explanation needs access to neighbors in the target class.
"""
needs_neighbours(ce::AbstractCounterfactualExplanation) = needs_neighbours(ce.generator)

"A base type for AD backend requirements"
abstract type ADRequirements end

"By default, no special AD backend is required."
struct NoADRequirements <: ADRequirements end

ADRequirements(::Type) = NoADRequirements()

"This trait implies that `ForwardDiff` is required."
struct NeedsForwardDiff <: ADRequirements end

"The `hinge_loss` function requires `ForwardDiff`"
ADRequirements(::Type{<:typeof(hinge_loss)}) = NeedsForwardDiff()

"The `energy_constraint` function requires `ForwardDiff`"
ADRequirements(::Type{<:typeof(energy_constraint)}) = NeedsForwardDiff()

choose_ad_backend(::NoADRequirements) = get_global_ad_backend()
choose_ad_backend(x::T) where {T} = choose_ad_backend(ADRequirements(T), x)
choose_ad_backend(::NoADRequirements, x) = get_global_ad_backend()
choose_ad_backend(::NeedsForwardDiff, x) = DI.AutoForwardDiff()

needs_special_backend(x::T) where {T} = needs_special_backend(ADRequirements(T), x)
needs_special_backend(::NoADRequirements, x) = false
needs_special_backend(::NeedsForwardDiff, x) = true

"Chooses the appropriate AD backend for the given model type"
choose_ad_backend(mod::Model) = choose_ad_backend(mod.type)

"""
    choose_ad_backend(gen::AbstractGenerator)

Choose an appropriate automatic differentiation backend for a given generator based on its penalty function. Handles both simple cases (no penalty function) and complex cases where multiple penalties might require different AD backends, ensuring that these backends are mutually exclusive.
"""
function choose_ad_backend(gen::AbstractGenerator)

    # Simplest case (no penalty function at all):
    if !hasfield(typeof(gen), :penalty)
        return choose_ad_backend(NoADRequirements())
    end

    # No penalty supplied:
    if isnothing(gen.penalty)
        return choose_ad_backend(NoADRequirements())
    end

    # Flatten the penalty to ensure it's a single object or array of penalties
    penalty = CounterfactualExplanations.flatten_penalty(gen.penalty)

    # Check if any penalty requires a special AD backend
    if any(needs_special_backend.(penalty))
        @assert length(unique(choose_ad_backend.(penalty))) <= 2 "You have specified two or more penalties that require mutually exclusive AD backends."

        # Filter penalties that need special backends
        penalty = penalty[findall(needs_special_backend.(penalty))]

        # Choose the unique backend among those required by the filtered penalties
        return choose_ad_backend.(penalty) |> x -> unique(x)[1]
    else
        # If no special backend is needed, use the default AD backend
        return choose_ad_backend(NoADRequirements())
    end
end

"""
    choose_ad_backend(backends::Vararg{<:Any})

Select a single automatic differentiation backend from multiple provided backends.

# Arguments
- `backends::Vararg{<:Any}`: Variable number of AD backend instances (e.g., `AutoZygote()`, `AutoEnzyme()`).

# Returns
- `AutoZygote()` if all provided backends are `AutoZygote` instances.
- The single non-`AutoZygote` backend if exactly one is provided among `AutoZygote` instances.

# Throws
- `AssertionError` if more than one non-`AutoZygote` backend is provided.

# Examples

```julia
choose_ad_backend(AutoZygote(), AutoZygote())  # Returns AutoZygote()
choose_ad_backend(AutoZygote(), AutoEnzyme())  # Returns AutoEnzyme()
choose_ad_backend(AutoZygote(), AutoEnzyme(), AutoForwardDiff())  # Throws AssertionError
```
"""
function choose_ad_backend(backends::Vararg{<:Any}; default_bkd=DI.AutoZygote())
    # Filter out non-default backends
    non_default = filter(b -> b != default_bkd, backends) |> unique

    # Assert only one non-Zygote backend
    @assert length(non_default) <= 1 "Only one non-default backend is allowed, got $(length(non_default))"

    # Return non-default backend if it exists, otherwise return default
    if isempty(non_default)
        return default_bkd
    else
        return first(non_default)
    end
end

"""
    choose_ad_backend(ce::CounterfactualExplanation)

Select a compatible automatic differentiation backend for a counterfactual explanation.

Determines the AD backend by querying the backends of the model and generator within the
counterfactual explanation, then reconciling them into a single compatible backend.

# Arguments
- `ce::CounterfactualExplanation`: A counterfactual explanation object containing a model (`M`) and a generator.

# Returns
- A single AD backend compatible with both the model and generator.

# Throws
- `AssertionError` if the model and generator use incompatible non-`AutoZygote` backends.

# See Also
- [`choose_ad_backend(backends::Vararg)`](@ref)
"""
function choose_ad_backend(ce::AbstractCounterfactualExplanation)
    n_features = CounterfactualExplanations.DataPreprocessing.input_dim(ce.data)

    # Check for model/generator requirments:
    bkd_mod = choose_ad_backend(ce.M)
    bkd_gen = choose_ad_backend(ce.generator)

    return choose_ad_backend(bkd_mod, bkd_gen)
end
