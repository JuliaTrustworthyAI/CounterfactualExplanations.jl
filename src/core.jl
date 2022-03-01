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
function generate_recourse(generator::Generator, x̅::AbstractArray, 𝑴::Models.FittedModel, target::Union{Float64,Int}, γ::Float64; T=1000, feasible_range=nothing)
    
    # Setup and allocate memory:
    x̲ = copy(x̅) # start from factual
    p̅ = Models.probs(𝑴, x̅)
    out_dim = size(p̅)[1]
    y̅ = out_dim == 1 ? round(p̅[1]) : Flux.onecold(p̅,1:out_dim)
    # If multi-class, onehot-encode target
    target_hot = out_dim > 1 ? Flux.onehot(target, 1:out_dim) : target
    D = length(x̲)
    path = [x̲]
    𝑷 = zeros(D) # vector to keep track of number of permutations by feature
    𝑭ₜ = initialize_mutability(generator, D) 

    # Initialize:
    t = 1 # counter
    not_finished = true # convergence condition

    # Search:
    while not_finished
        # println(t)
        # Generate peturbations
        Δx̲ = Generators.generate_perturbations(generator, x̲, 𝑴, target_hot, x̅, 𝑭ₜ)
        𝑭ₜ = Generators.mutability_constraints(generator, 𝑭ₜ, 𝑷) # generate mutibility constraint mask
        Δx̲ = reshape(apply_mutability(Δx̲, 𝑭ₜ), size(x̲)) # apply mutability constraints
        
        # Updates:
        x̲ += Δx̲ # update counterfactual
        if !isnothing(feasible_range)
            clamp!(x̲, feasible_range[1], feasible_range[2])
        end
        path = [path..., x̲]
        𝑷 += reshape(Δx̲ .!= 0, size(𝑷)) # update number of times feature has been changed
        t += 1 # update iteration counter
        global converged = threshold_reached(𝑴, x̲, target, γ)
        not_finished = t < T && !converged && !Generators.conditions_satisified(generator, x̲, 𝑴, target, x̅, 𝑷)

    end

    # Output:
    p̲ = Models.probs(𝑴, x̲)
    y̲ = out_dim == 1 ? round(p̲[1]) : Flux.onecold(p̲,1:out_dim)
    recourse = Recourse(x̲, y̲, p̲, path, generator, x̅, y̅, p̅, 𝑴, target, converged) 
    
    return recourse
    
end

"""
    target_probs(p, target)

Selects the probabilities of the target class. In case of binary classification problem `p` reflects the probability that `y=1`. In that case `1-p` reflects the probability that `y=0`.

# Examples

```julia-repl
using AlgorithmicRecourse
using AlgorithmicRecourse.Models: LogisticModel, probs 
Random.seed!(1234)
N = 25
w = [1.0 1.0]# true coefficients
b = 0
x, y = toy_data_linear(N)
# Logit model:
𝑴 = LogisticModel(w, [b])
p = probs(𝑴, x[rand(N)])
target_probs(p, 0)
target_probs(p, 1)
```

"""
function target_probs(p, target)
    if size(p)[1] == 1
        # If target is binary (i.e. outcome 1D from sigmoid), compute p(y=0):
        p = vcat(1.0 .- p, p)
        # Choose first (target+1) row if target=0, second row (target+1) if target=1:  
        p_target = p[Int(target+1),:]
    else
        # If target is multi-class, choose corresponding row (e.g. target=2 -> row 2)
        p_target = p[Int(target),:]
    end
    return p_target
end

"""
    threshold_reached(𝑴::Models.FittedModel, x̲::AbstractArray, target::Float64, γ::Float64)

Checks if confidence threshold has been reached. 
"""
threshold_reached(𝑴::Models.FittedModel, x̲::AbstractArray, target::Real, γ::Real) = target_probs(Models.probs(𝑴, x̲), target)[1] >= γ

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
    y̲::Union{Real,AbstractArray}
    p̲::Any
    path::AbstractArray
    generator::Generators.Generator
    x̅::AbstractArray
    y̅::Union{Real,AbstractArray}
    p̅::Any
    𝑴::Models.FittedModel
    target::Real
    converged::Bool
end;
