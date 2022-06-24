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
    x::Union{AbstractArray,Int}, target::Union{AbstractFloat,Int}, data::CounterfactualData, M::Models.AbstractFittedModel, generator::AbstractGenerator;
    γ::AbstractFloat=0.75, T::Int=1000, latent_space::Union{Nothing, Bool}=nothing
)
    # Initialize:
    counterfactual = CounterfactualExplanation(x=x, target=target, data=data, M=M, generator=generator, γ=γ, T=T, latent_space=latent_space)

    # Search:
    while !counterfactual.search[:terminated]
        update!(counterfactual)
    end

    return counterfactual
    
end