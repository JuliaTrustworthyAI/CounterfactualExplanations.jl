# core.jl

# -------- Main method:
"""
    generate_recourse(generator::Generator, x̅::AbstractArray, 𝑴::Models.FittedModel, target::Float64; T=1000, 𝓘=[])

Takes a recourse `generator`, the factual sample `x̅`, the fitted model `𝑴` and the `target` label. Returns the generated recourse (an object of type `Recourse`).

# Examples

```julia-repl
w = reshape([1.0,-2.0],2,1) # true coefficients
b = [0]
x̅ = [-1,0.5]
target = 1.0
𝑴 = AlgorithmicRecourse.Models.LogisticModel(w, b);
generator = GenericGenerator(0.1,0.1,1e-5)
recourse = generate_recourse(generator, x̅, 𝑴, target); # generate recourse
```

See also [`GenericGenerator(λ::Float64, ϵ::Float64, τ::Float64)`](@ref)
"""

function generate_recourse(generator::Generator, x̅::AbstractArray, 𝑴::Models.FittedModel, target::Float64; T=1000, 𝓘=[])
    
    # Setup and allocate memory:
    x̲ = copy(x̅) # start from factual
    D = length(x̲)
    path = reshape(x̲, 1, length(x̲)) # storing the path

    # Initialize:
    t = 1 # counter
    converged = Generators.convergence(generator, x̲, 𝑴, target, x̅) 

    # Search:
    while !converged && t < T 
        x̲ = Generators.step(generator, x̲, 𝑴, target, x̅, 𝓘)
        t += 1 # update number of times feature is changed
        converged = Generators.convergence(generator, x̲, 𝑴, target, x̅) # check if converged
        path = vcat(path, reshape(x̲, 1, D))
    end

    # Output:
    y̲ = round.(Models.probs(𝑴, x̲))[1]
    recourse = Recourse(x̲, y̲, path, generator, 𝓘, x̅, 𝑴, target) 
    
    return recourse
    
end

struct Recourse
    x̲::AbstractArray
    y̲::Float64
    path::Matrix{Float64}
    generator::Generators.Generator
    immutable::AbstractArray
    x̅::AbstractArray
    𝑴::Models.FittedModel
    target::Float64
end;

# --------------- Outer constructor methods: 
# Check if recourse is valid:
function valid(recourse::Recourse; 𝑴=nothing)
    if isnothing(𝑴)
        valid = recourse.y̲ == recourse.target
    else 
        valid = 𝑴(recourse.x̲) == recourse.target
    end
    return valid
end

# Compute cost associated with counterfactual:
function cost(recourse::Recourse; cost_fun=nothing, cost_fun_kargs)
    return cost_fun(recourse.generator.x̅, recourse.x̲; cost_fun_kargs...)
end