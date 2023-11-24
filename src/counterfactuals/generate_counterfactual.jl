# -------- Main method:
"""
	generate_counterfactual(
		x::Union{AbstractArray,Int}, target::RawTargetType, data::CounterfactualData, M::Models.AbstractFittedModel, generator::AbstractGenerator;
		γ::AbstractFloat=0.75, max_iter=1000
	)

The core function that is used to run counterfactual search for a given factual `x`, target, counterfactual data, model and generator. 
Keywords can be used to specify the desired threshold for the predicted target class probability and the maximum number of iterations.

# Examples

## Generic generator

```julia-repl
using CounterfactualExplanations

# Data:
using CounterfactualExplanations.Data
using Random
Random.seed!(1234)
xs, ys = Data.toy_data_linear()
X = hcat(xs...)
counterfactual_data = CounterfactualData(X,ys')

# Model
using CounterfactualExplanations.Models: LogisticModel, probs 
# Logit model:
w = [1.0 1.0] # true coefficients
b = 0
M = LogisticModel(w, [b])

# Randomly selected factual:
x = select_factual(counterfactual_data,rand(1:size(X)[2]))
y = round(probs(M, x)[1])
target = round(probs(M, x)[1])==0 ? 1 : 0 

# Counterfactual search:
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
```
"""
function generate_counterfactual(
    x::AbstractArray,
    target::RawTargetType,
    data::CounterfactualData,
    M::Models.AbstractFittedModel,
    generator::AbstractGenerator;
    num_counterfactuals::Int=1,
    initialization::Symbol=:add_perturbation,
    max_iter::Int=100,
    decision_threshold::AbstractFloat=0.5,
    gradient_tol::AbstractFloat=parameters[:τ],
    min_success_rate::AbstractFloat=parameters[:min_success_rate],
    converge_when::Symbol=:decision_threshold,
    timeout::Union{Nothing,Int}=nothing,
)
    # Initialize:
    ce = CounterfactualExplanation(
        x,
        target,
        data,
        M,
        generator;
        num_counterfactuals=num_counterfactuals,
        initialization=initialization,
        max_iter=max_iter,
        min_success_rate=min_success_rate,
        decision_threshold=decision_threshold,
        gradient_tol=gradient_tol,
        converge_when=converge_when,
    )

    # Search:
    if isa(generator, AbstractGradientBasedGenerator)
        timer = isnothing(timeout) ? nothing : Timer(timeout)
        while !ce.search[:terminated]
            update!(ce)
            if !isnothing(timer)
                yield()
                if !isopen(timer)
                    @info "Counterfactual search timed out before convergence"
                    break
                end
            end
        end

    elseif isa(generator, FeatureTweakGenerator)

        # Asserts related to https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl/issues/258
        @assert ce.data.standardize == false "The `FeatureTweakGenerator` currently doesn't support feature encodings."
        @assert ce.generator.latent_space == false "The `FeatureTweakGenerator` currently doesn't support feature encodings."

        if isa(M, Models.TreeModel)
            ce = Generators.feature_tweaking!(ce)
            ce.search[:path] = [ce.search[:path]..., ce.s′]
            ce.search[:iteration_count] = 1
            ce.search[:terminated] = true
            ce.search[:converged] = true
            ce.x′ = decode_state(ce)                                    # decoded counterfactual state
        else
            @warn "The `FeatureTweakGenerator` currently only supports tree models. The counterfactual search will be terminated."
        end

    elseif isa(generator, GrowingSpheresGenerator)

        # Asserts related to https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl/issues/258
        @assert ce.data.standardize == false "The `GrowingSpheres` currently doesn't support feature encodings."
        @assert ce.generator.latent_space == false "The `GrowingSpheres` currently doesn't support feature encodings."

        Generators.growing_spheres_generation!(ce)
        Generators.feature_selection!(ce)
        ce.x′ = decode_state(ce)                                    # decoded counterfactual state
    else
        @error "Generator not recognized."
    end
    return ce
end

"Overloads the `generate_counterfactual` method to accept a tuple containing and array. This allows for broadcasting over `Zip` iterators."
function generate_counterfactual(x::Tuple{<:AbstractArray}, args...; kwargs...)
    return generate_counterfactual(x[1], args...; kwargs...)
end
