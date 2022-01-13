# core.jl

# -------- Main method:
"""
    generate_recourse(generator::Generator, x̅::AbstractArray, 𝓜::Models.FittedModel, target::Float64; T=1000, 𝓘=[])

Takes a recourse `generator`, the factual sample `x̅`, the fitted model `𝓜` and the `target` label. Returns the generated recourse (an object of type `Recourse`).

# Examples

## Generic generators

```julia-repl
w = [1.0,-2.0] # true coefficients
b = [0]
x̅ = reshape([-1,0.5],1,2)
target = 1.0
𝓜 = AlgorithmicRecourse.Models.LogisticModel(w, b);
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy)
recourse = generate_recourse(generator, x̅, 𝓜, target); # generate recourse
```

## Greedy generator for Bayesian model

```julia-repl
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0, 1.0,-2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
x̅ = reshape([-1,0.5],1,2)
target = 1.0
𝓜 = AlgorithmicRecourse.Models.BayesianLogisticModel(μ, Σ);
generator = GreedyGenerator(0.95,0.01,20,:logitbinarycrossentropy)
recourse = generate_recourse(generator, x̅, 𝓜, target); # generate recourse
```

See also [`GenericGenerator(λ::Float64, ϵ::Float64, τ::Float64)`](@ref), [`GreedyGenerator(Γ::Float64, δ::Float64, n::Int64, loss::Symbol)`](@ref).
"""
function generate_recourse(generator::Generator, x̅::AbstractArray, 𝓜::Models.FittedModel, target::Float64; T=1000, 𝓘=[])
    
    # Setup and allocate memory:
    x̲ = copy(x̅) # start from factual
    y̅ = round.(Models.probs(𝓜, x̅))[1]
    D = length(x̲)
    path = reshape(x̲, 1, length(x̲)) # storing the path

    # Initialize:
    t = 1 # counter
    converged = Generators.convergence(generator, x̲, 𝓜, target, x̅) 

    # Search:
    while !converged && t < T 
        x̲ = Generators.step(generator, x̲, 𝓜, target, x̅, 𝓘)
        t += 1 # update number of times feature is changed
        converged = Generators.convergence(generator, x̲, 𝓜, target, x̅) # check if converged
        path = vcat(path, reshape(x̲, 1, D))
    end

    # Output:
    y̲ = round.(Models.probs(𝓜, x̲))[1]
    recourse = Recourse(x̲, y̲, path, generator, 𝓘, x̅, y̅, 𝓜, target) 
    
    return recourse
    
end

"""
    Recourse(
        x̲::AbstractArray
        y̲::Float64
        path::Matrix{Float64}
        generator::Generators.Generator
        𝓘::AbstractArray
        x̅::AbstractArray
        y̅::Float64
        𝓜::Models.FittedModel
        target::Float64
    )

Collects all variables relevant to the recourse outcome. 
"""
struct Recourse
    x̲::AbstractArray
    y̲::Float64
    path::Matrix{Float64}
    generator::Generators.Generator
    𝓘::AbstractArray
    x̅::AbstractArray
    y̅::Float64
    𝓜::Models.FittedModel
    target::Float64
end;

# # --------------- Outer constructor methods: 
# # Check if recourse is valid:
# function valid(recourse::Recourse; 𝓜=nothing)
#     if isnothing(𝓜)
#         valid = recourse.y̲ == recourse.target
#     else 
#         valid = 𝓜(recourse.x̲) == recourse.target
#     end
#     return valid
# end

# # Compute cost associated with counterfactual:
# function cost(recourse::Recourse; cost_fun=nothing, cost_fun_kargs)
#     return cost_fun(recourse.generator.x̅, recourse.x̲; cost_fun_kargs...)
# end