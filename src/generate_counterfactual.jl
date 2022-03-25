# -------- Main method:
"""
    generate_counterfactual(generator::AbstractGenerator, x̅::Vector, 𝑴::Models.AbstractFittedModel, target::AbstractFloat, γ::AbstractFloat; T=1000)

Takes a recourse `generator`, the factual sample `x̅`, the fitted model `𝑴`, the `target` label and its desired threshold probability `γ`. Returns the generated recourse (an object of type `Recourse`).

# Examples

## Generic generator

```julia-repl
using CounterfactualExplanations.Models
w = [1.0 -2.0] # true coefficients
b = [0]
x̅ = [-1,0.5]
target = 1.0
γ = 0.9
𝑴 = LogisticModel(w, b)
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ); # generate recourse
```

## Greedy generator (Bayesian model only)

```julia-repl
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0 1.0 -2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
x̅ = [-1,0.5]
target = 1.0
γ = 0.9
𝑴 = CounterfactualExplanations.Models.BayesianLogisticModel(μ, Σ);
generator = GreedyGenerator(0.01,20,:logitbinarycrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ); # generate recourse
```

See also:

- [`GenericGenerator(λ::AbstractFloat, ϵ::AbstractFloat, τ::AbstractFloat, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})`](@ref)
- [`GreedyGenerator(δ::AbstractFloat, n::Int64, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})`](@ref).
"""
function generate_counterfactual(
    x̅::Union{AbstractArray,Int}, target::Union{AbstractFloat,Int}, data::CounterfactualData, 𝑴::Models.AbstractFittedModel, generator::AbstractGenerator;
    γ::AbstractFloat=0.75, T=1000, feasible_range=nothing
)
    # Initialize:
    counterfactual = CounterfactualExplanation(x̅, target, data, 𝑴, generator, γ, T)
    initialize!(counterfactual) 

    # Search:
    while !counterfactual.search[:terminated]
        update!(counterfactual)
    end

    return counterfactual
    
end