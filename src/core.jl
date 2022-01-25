# core.jl

# -------- Main method:
"""
    generate_recourse(generator::Generator, x̅::Vector, 𝑴::Models.FittedModel, target::Float64, γ::Float64; T=1000)

Takes a recourse `generator`, the factual sample `x̅`, the fitted model `𝑴`, the `target` label and its desired threshold probability `γ`. Returns the generated recourse (an object of type `Recourse`).

# Examples

## Generic generator

```julia-repl
w = [1.0 -2.0] # true coefficients
b = [0]
x̅ = [-1,0.5]
target = 1.0
γ = 0.9
𝑴 = AlgorithmicRecourse.Models.LogisticModel(w, b);
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)
recourse = generate_recourse(generator, x̅, 𝑴, target, γ); # generate recourse
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
𝑴 = AlgorithmicRecourse.Models.BayesianLogisticModel(μ, Σ);
generator = GreedyGenerator(0.01,20,:logitbinarycrossentropy,nothing)
recourse = generate_recourse(generator, x̅, 𝑴, target, γ); # generate recourse
```

See also:

- [`GenericGenerator(λ::Float64, ϵ::Float64, τ::Float64, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})`](@ref)
- [`GreedyGenerator(δ::Float64, n::Int64, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})`](@ref).
"""
function generate_recourse(generator::Generator, x̅::Vector, 𝑴::Models.FittedModel, target::Float64, γ::Float64; T=1000)
    
    # Setup and allocate memory:
    x̲ = copy(x̅) # start from factual
    y̅ = round.(Models.probs(𝑴, x̅))[1]
    D = length(x̲)
    path = reshape(x̲, 1, length(x̲)) # storing the path
    𝑷 = zeros(D) # vector to keep track of number of permutations by feature
    𝑭ₜ = initialize_mutability(generator, D) 

    # Initialize:
    t = 1 # counter
    not_converged = true # convergence condition

    # Search:
    while not_converged

        # Generate peturbations
        Δx̲ = Generators.generate_perturbations(generator, x̲, 𝑴, target, x̅, 𝑭ₜ)
        𝑭ₜ = Generators.mutability_constraints(generator, 𝑭ₜ, 𝑷) # generate mutibility constraint mask
        Δx̲ = reshape(apply_mutability(Δx̲, 𝑭ₜ), size(x̲)) # apply mutability constraints
        
        # Updates:
        x̲ += Δx̲ # update counterfactual
        path = vcat(path, reshape(x̲, 1, D)) # update counterfactual path
        𝑷 += reshape(Δx̲ .!= 0, size(𝑷)) # update number of times feature has been changed
        t += 1 # update iteration counter
        not_converged = t < T && !threshold_reached(𝑴, x̲, target, γ) && !Generators.conditions_satisified(generator, x̲, 𝑴, target, x̅, 𝑷)

    end

    # Output:
    y̲ = round.(Models.probs(𝑴, x̲))[1]
    recourse = Recourse(x̲, y̲, path, generator, x̅, y̅, 𝑴, target) 
    
    return recourse
    
end

"""
    threshold_reached(𝑴::Models.FittedModel, x̲::AbstractArray, target::Float64, γ::Float64)

Checks if confidence threshold has been reached. 
"""
threshold_reached(𝑴::Models.FittedModel, x̲::AbstractArray, target::Float64, γ::Float64) = abs(Models.probs(𝑴, x̲)[1] - target) <= abs(target-γ)

"""
    apply_mutability(Δx̲::AbstractArray, 𝑭::Vector{Symbol})

Apply mutability constraints to `Δx̲` based on vector of constraints `𝑭`.

# Examples 

𝑭 = [:both, :increase, :decrease, :none]
_mutability([-1,1,-1,1], 𝑭) # all but :none pass
_mutability([-1,-1,-1,1], 𝑭) # all but :increase and :none pass
_mutability([-1,1,1,1], 𝑭) # all but :decrease and :none pass
_mutability([-1,-1,1,1], 𝑭) # only :both passes

"""
function apply_mutability(Δx̲::AbstractArray, 𝑭::Vector{Symbol})

    both(x) = x
    increase(x) = ifelse(x<0,0,x)
    decrease(x) = ifelse(x>0,0,x)
    none(x) = 0

    cases = (both = both, increase = increase, decrease = decrease, none = none)

    Δx̲ = [getfield(cases, 𝑭[d])(Δx̲[d]) for d in 1:length(Δx̲)]

    return Δx̲

end

function initialize_mutability(generator::Generator, D::Int64)
    if isnothing(generator.𝑭)
        𝑭 = [:both for i in 1:D]
    else 
        𝑭 = generator.𝑭
    end
    return 𝑭
end

"""
    Recourse(x̲::AbstractArray, y̲::Float64, path::Matrix{Float64}, generator::Generators.Generator, x̅::AbstractArray, y̅::Float64, 𝑴::Models.FittedModel, target::Float64)

Collects all variables relevant to the recourse outcome. 
"""
struct Recourse
    x̲::AbstractArray
    y̲::Float64
    path::Matrix{Float64}
    generator::Generators.Generator
    x̅::AbstractArray
    y̅::Float64
    𝑴::Models.FittedModel
    target::Float64
end;

# # --------------- Outer constructor methods: 
# # Check if recourse is valid:
# function valid(recourse::Recourse; 𝑴=nothing)
#     if isnothing(𝑴)
#         valid = recourse.y̲ == recourse.target
#     else 
#         valid = 𝑴(recourse.x̲) == recourse.target
#     end
#     return valid
# end

# # Compute cost associated with counterfactual:
# function cost(recourse::Recourse; cost_fun=nothing, cost_fun_kargs)
#     return cost_fun(recourse.generator.x̅, recourse.x̲; cost_fun_kargs...)
# end