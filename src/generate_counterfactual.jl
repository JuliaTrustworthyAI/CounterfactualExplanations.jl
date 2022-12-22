# -------- Main method:
"""
    generate_counterfactual(
        x::Union{AbstractArray,Int}, target::Union{AbstractFloat,Int}, data::CounterfactualData, M::Models.AbstractFittedModel, generator::AbstractGenerator;
        γ::AbstractFloat=0.75, T=1000
    )

The core function that is used to run counterfactual search for a given factual `x`, target, counterfactual data, model and generator. Keywords can be used to specify the desired threshold for the predicted target class probability and the maximum number of iterations.

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
counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator)
```
"""
function generate_counterfactual(
    x::AbstractArray,
    target::Union{AbstractFloat,Int},
    data::CounterfactualData,
    M::Models.AbstractFittedModel,
    generator::AbstractGenerator;
    T::Int = 1000,
    latent_space::Union{Nothing,Bool} = nothing,
    num_counterfactuals::Int = 1,
    initialization::Symbol = :add_perturbation,
    generative_model_params::NamedTuple = (;),
)
    # Initialize:
    counterfactual = CounterfactualExplanation(
        x = x,
        target = target,
        data = data,
        M = M,
        generator = generator,
        T = T,
        latent_space = latent_space,
        num_counterfactuals = num_counterfactuals,
        initialization = initialization,
        generative_model_params = generative_model_params,
    )

    # Search:
    while !counterfactual.search[:terminated]
        update!(counterfactual)
    end

    return counterfactual

end


function generate_counterfactual(
    x::Base.Iterators.Zip,
    target::Union{AbstractFloat,Int},
    data::CounterfactualData,
    M::Models.AbstractFittedModel,
    generator::AbstractGenerator;
    kwargs...,
)
    counterfactuals =
        map(x_ -> generate_counterfactual(x_[1], target, data, M, generator; kwargs...), x)

    return counterfactuals

end
