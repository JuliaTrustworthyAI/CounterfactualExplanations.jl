# core.jl

# -------- Main method:
function generate_recourse(generator::Generator, x̅::AbstractArray, 𝓜::Models.FittedModel, target::Float64; T=1000, 𝓘=[])
    
    # Setup and allocate memory:
    x̲ = copy(x̅) # start from factual
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
    recourse = Recourse(x̲, y̲, path, generator, 𝓘, x̅, 𝓜, target) 
    
    return recourse
    
end

struct Recourse
    x̲::AbstractArray
    y̲::Float64
    path::Matrix{Float64}
    generator::Generators.Generator
    immutable::AbstractArray
    x̅::AbstractArray
    𝓜::Models.FittedModel
    target::Float64
end;

# --------------- Outer constructor methods: 
# Check if recourse is valid:
function valid(recourse::Recourse; 𝓜=nothing)
    if isnothing(𝓜)
        valid = recourse.y̲ == recourse.target
    else 
        valid = 𝓜(recourse.x̲) == recourse.target
    end
    return valid
end

# Compute cost associated with counterfactual:
function cost(recourse::Recourse; cost_fun=nothing, cost_fun_kargs)
    return cost_fun(recourse.generator.x̅, recourse.x̲; cost_fun_kargs...)
end