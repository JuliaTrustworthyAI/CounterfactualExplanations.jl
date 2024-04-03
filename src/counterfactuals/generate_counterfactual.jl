"""
	generate_counterfactual(
        x::Matrix,
        target::RawTargetType,
        data::CounterfactualData,
        M::Models.AbstractFittedModel,
        generator::AbstractGenerator;
        num_counterfactuals::Int=1,
        initialization::Symbol=:add_perturbation,
        convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
        timeout::Union{Nothing,Real}=nothing,
    )

The core function that is used to run counterfactual search for a given factual `x`, target, counterfactual data, model and generator. Keywords can be used to specify the desired threshold for the predicted target class probability and the maximum number of iterations.

# Arguments

- `x::Matrix`: Factual data point.
- `target::RawTargetType`: Target class.
- `data::CounterfactualData`: Counterfactual data.
- `M::Models.AbstractFittedModel`: Fitted model.
- `generator::AbstractGenerator`: Generator.
- `num_counterfactuals::Int=1`: Number of counterfactuals to generate for factual.
- `initialization::Symbol=:add_perturbation`: Initialization method. By default, the initialization is done by adding a small random perturbation to the factual to achieve more robustness.
- `convergence::Union{AbstractConvergence,Symbol}=:decision_threshold`: Convergence criterion. By default, the convergence is based on the decision threshold.
- `timeout::Union{Nothing,Int}=nothing`: Timeout in seconds.

# Examples

## Generic generator

```julia
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
    x::Matrix,
    target::RawTargetType,
    data::CounterfactualData,
    M::Models.AbstractFittedModel,
    generator::AbstractGenerator;
    num_counterfactuals::Int=1,
    initialization::Symbol=:add_perturbation,
    convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
    timeout::Union{Nothing,Real}=nothing,
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
        convergence=convergence,
    )

    # Search:
    timer = isnothing(timeout) ? nothing : Timer(timeout)
    while !terminated(ce)
        update!(ce)
        if !isnothing(timer)
            yield()
            if !isopen(timer)
                @info "Counterfactual search timed out before convergence"
                break
            end
        end
    end
    return ce
end

"Overloads the `generate_counterfactual` method to accept a tuple containing and array. This allows for broadcasting over `Zip` iterators."
function generate_counterfactual(x::Tuple{<:AbstractArray}, args...; kwargs...)
    return generate_counterfactual(x[1], args...; kwargs...)
end
