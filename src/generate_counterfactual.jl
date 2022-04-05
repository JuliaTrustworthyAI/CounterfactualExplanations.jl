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

# Raw Data:
using CounterfactualExplanations.Data: cats_dogs_data
x, y = cats_dogs_data()

# Data preprocessing:
counterfactual_data = CounterfactualData(x,y)

# Model (pre-trained):
using CounterfactualExplanations.Data: cats_dogs_laplace
import CounterfactualExplanations.Models: probs
la = cats_dogs_model()

# Counterfactual search:
x = select_factual(counterfactual_data, 1) # factual
target = round(probs(la, x)) == 1.0 ? 0.0 : 1.0
generator = GenericGenerator()
counterfactual = generate_counterfactual(x, target, counterfactual_data, la, generator)
```

## Greedy generator (Bayesian model only)

```julia-repl
using CounterfactualExplanations

# Raw Data:
using CounterfactualExplanations.Data: cats_dogs_data
x, y = cats_dogs_data()

# Data preprocessing:
counterfactual_data = CounterfactualData(x,y)

# Model (pre-trained):
using CounterfactualExplanations.Data: cats_dogs_laplace
import CounterfactualExplanations.Models: probs
la = cats_dogs_laplace()

# Counterfactual search:
x = select_factual(counterfactual_data, 1) # factual
target = round(probs(la, x)) == 1.0 ? 0.0 : 1.0
generator = GreedyGenerator()
counterfactual = generate_counterfactual(x, target, counterfactual_data, la, generator)
```
"""
function generate_counterfactual(
    x::Union{AbstractArray,Int}, target::Union{AbstractFloat,Int}, data::CounterfactualData, M::Models.AbstractFittedModel, generator::AbstractGenerator;
    γ::AbstractFloat=0.75, T=1000
)
    # Initialize:
    counterfactual = CounterfactualExplanation(x, target, data, M, generator, γ, T)
    initialize!(counterfactual) 

    # Search:
    while !counterfactual.search[:terminated]
        update!(counterfactual)
    end

    return counterfactual
    
end