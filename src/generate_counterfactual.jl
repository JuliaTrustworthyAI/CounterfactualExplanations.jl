# -------- Main method:
"""
    generate_counterfactual(generator::AbstractGenerator, x::Vector, M::Models.AbstractFittedModel, target::AbstractFloat, γ::AbstractFloat; T=1000)

Takes a recourse `generator`, the factual sample `x`, the fitted model `M`, the `target` label and its desired threshold probability `γ`. Returns the generated recourse (an object of type `Recourse`).

# Examples

## Generic generator

```julia-repl
using CounterfactualExplanations.Models
w = [1.0 -2.0] # true coefficients
b = [0]
x = [-1,0.5]
target = 1.0
γ = 0.9
M = LogisticModel(w, b)
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)
counterfactual = generate_counterfactual(generator, x, M, target, γ); # generate recourse
```

## Greedy generator (Bayesian model only)

```julia-repl
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0 1.0 -2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
x = [-1,0.5]
target = 1.0
γ = 0.9
M = CounterfactualExplanations.Models.BayesianLogisticModel(μ, Σ);
generator = GreedyGenerator(0.01,20,:logitbinarycrossentropy,nothing)
counterfactual = generate_counterfactual(generator, x, M, target, γ); # generate recourse
```

See also:

- [`GenericGenerator(λ::AbstractFloat, ϵ::AbstractFloat, τ::AbstractFloat, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})`](@ref)
- [`GreedyGenerator(δ::AbstractFloat, n::Int64, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})`](@ref).
"""
function generate_counterfactual(
    x::Union{AbstractArray,Int}, target::Union{AbstractFloat,Int}, data::CounterfactualData, M::Models.AbstractFittedModel, generator::AbstractGenerator;
    γ::AbstractFloat=0.75, T=1000, feasible_range=nothing
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