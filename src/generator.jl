# generators.jl
#
# Core package functionality that implements algorithmic recourse.

# --------------- Core constructor:
abstract type Generator end

# --- Outer methods:

# Generate recourse:
function generate_recourse(generator::Generator, x̅::Vector{x}, 𝓜, target::Float64; T=1000, immutable_=[])
    
    # Setup and allocate memory:
    x̲ = copy(x̅) # start from factual
    D = length(x̲)
    path = reshape(x̲, 1, length(x̲)) # storing the path

    # Initialize:
    t = 1 # counter
    converged = convergence(generator, x̅, 𝓜, target, x̲)

    # Search:
    while !converged && t < T 
        x̲ = step(generator, x̅, 𝓜, target, x̲)
        t += 1 # update number of times feature is changed
        converged = convergence(generator, x̅, 𝓜, target, x̲) # check if converged
        path = vcat(path, reshape(x̲, 1, D))
    end

    # Output:
    y̲ = 𝓜(x̲)
    recourse = Recourse(x̲, y̲, path, generator, immutable_, x̅, 𝓜, target) 
    
    return recourse
    
end

# Generators:
struct GenericGenerator <: Generator
    λ::Float64
    ϵ::Float64
    τ::Float64
end

ℓ(generator::GenericGenerator, 𝓜, t) = - (t * log(𝛔(a)) + (1-t) * log(1-𝛔(a)))
cost(generator::GenericGenerator, x̅, x̲) = norm(x̅,x̲)^2
objective(generator::GenericGenerator, x̅, 𝓜, target, x̲) = ℓ(generator, a, t) + generator.λ * cost(generator, x̅, x̲) 
∇(generator::GenericGenerator, x̅, 𝓜, target, x̲) = gradient(() -> objective(generator, a, t, x̅, x̲), params(x̲))

function step(generator::GenericGenerator, x̅, 𝓜, target, x̲) 
    𝐠ₜ = ∇(generator, x̅, 𝓜, target, x̲)
    return x̲ - (generator.ϵ .* 𝐠ₜ)
end

function convergence(generator::GenericGenerator, x̅, 𝓜, target, x̲)
    all(∇(generator, x̅, 𝓜, target, x̲) .< generator.τ)
end

# --------------- Wachter et al (2018): 
function gradient_cost(x_f, x̲)
    (x̲ .- x_f) ./ norm(x̲ .- x_f)
end;

function generate_recourse_wachter(x, gradient, classifier, target; T=1000, immutable_=[], a=1, τ=1e-5, λ=0.1, gradient_cost=gradient_cost)
    # Setup:
    w = coef(classifier)
    constant_needed = length(w) > length(x) # is adjustment for constant needed?
    if (constant_needed)
        x = vcat(1, x)
        immutable_ = vcat(1, immutable_ .+ 1) # adjust mask for immutable features
    end
    x̲ = copy(x) # start from factual
    D = length(x̲)
    path = reshape(x, 1, length(x)) # storing the path
    function convergence_condition(x̲, gradient, w, target, τ)
        all(gradient(x̲,w,target) .<= τ)
    end
    
    # Initialize:
    t = 1 # counter
    converged = convergence_condition(x̲, gradient, w, target, τ)
    
    # Recursion:
    while !converged && t < T 
        𝐠ₜ = gradient(x̲,w,target) # compute gradient
        𝐠ₜ[immutable_] .= 0 # set gradient of immutable features to zero
        g_cost_t = gradient_cost(x,x̲) # compute gradient of cost function
        g_cost_t[immutable_] .= 0 # set gradient of immutable features to zero
        cost = norm(x̲-x) # update cost
        if cost != 0
            x̲ -= (a .* (𝐠ₜ + λ .* g_cost_t)) # counterfactual update
        else
            x̲ -= (a .* 𝐠ₜ)
        end
        t += 1 # update number of times feature is changed
        converged = convergence_condition(x̲, gradient, w, target, τ) # check if converged
        path = vcat(path, reshape(x̲, 1, D))
    end
    
    # Output:
    y̲ = predict(classifier, x̲; proba=false)[1]
    valid = y̲ == target * 1.0
    cost = norm(x.-x̲)
    if (constant_needed)
        path = path[:,2:end]
        x̲ = x̲[2:end]
        x = x[2:end]
    end
    recourse = Recourse(x̲, y̲, path, target, valid, cost, x) 
    
    return recourse
end;

# --------------- Upadhyay et al (2021) 
function generate_recourse_roar(x, gradient, classifier, target; T=1000, immutable_=[], a=1, τ=1e-5, λ=0.1, gradient_cost=gradient_cost)
    # Setup:
    w = coef(classifier)
    constant_needed = length(w) > length(x) # is adjustment for constant needed?
    if (constant_needed)
        x = vcat(1, x)
        immutable_ = vcat(1, immutable_ .+ 1) # adjust mask for immutable features
    end
    x̲ = copy(x) # start from factual
    D = length(x̲)
    path = reshape(x, 1, length(x)) # storing the path
    function convergence_condition(x̲, gradient, w, target, tol)
        all(gradient(x̲,w,target) .<= τ)
    end
    
    # Initialize:
    t = 1 # counter
    converged = convergence_condition(x̲, gradient, w, target, τ)
    
    # Recursion:
    while !converged && t < T 
        𝐠ₜ = gradient(x̲,w,target) # compute gradient
        𝐠ₜ[immutable_] .= 0 # set gradient of immutable features to zero
        g_cost_t = gradient_cost(x,x̲) # compute gradient of cost function
        g_cost_t[immutable_] .= 0 # set gradient of immutable features to zero
        cost = norm(x̲-x) # update cost
        if cost != 0
            x̲ -= (a .* (𝐠ₜ + λ .* g_cost_t)) # counterfactual update
        else
            x̲ -= (a .* 𝐠ₜ)
        end
        t += 1 # update number of times feature is changed
        converged = convergence_condition(x̲, gradient, w, target, τ) # check if converged
        path = vcat(path, reshape(x̲, 1, D))
    end
    
    # Output:
    y̲ = predict(classifier, x̲; proba=false)[1]
    valid = y̲ == target * 1.0
    cost = norm(x.-x̲)
    if (constant_needed)
        path = path[:,2:end]
        x̲ = x̲[2:end]
        x = x[2:end]
    end
    recourse = Recourse(x̲, y̲, path, target, valid, cost, x) 
    
    return recourse
end;

# --------------- Schut et al (2021) 
function generate_recourse_schut(x,gradient,classifier,target;T=1000,immutable_=[],Γ=0.95,δ=1,n=nothing)
    # Setup:
    w = coef(classifier)
    constant_needed = length(w) > length(x) # is adjustment for constant needed?
    if (constant_needed)
        x = vcat(1, x)
        immutable_ = vcat(1, immutable_ .+ 1) # adjust mask for immutable features
    end
    x̲ = copy(x) # start from factual
    D = length(x̲)
    D_mutable = length(setdiff(1:D, immutable_))
    path = reshape(x, 1, length(x)) # storing the path
    if isnothing(n)
        n = ceil(T/D_mutable)
    end
    
    # Intialize:
    t = 1 # counter
    P = zeros(D) # number of times feature is changed
    converged = posterior_predictive(classifier, x̲)[1] .> Γ # check if converged
    max_number_changes_reached = all(P.==n)
    
    # Recursion:
    while !converged && t < T && !max_number_changes_reached
        𝐠ₜ = gradient(x̲,w,target) # compute gradient
        𝐠ₜ[P.==n] .= 0 # set gradient to zero, if already changed n times 
        𝐠ₜ[immutable_] .= 0 # set gradient of immutable features to zero
        i_t = argmax(abs.(𝐠ₜ)) # choose most salient feature
        x̲[i_t] -= δ * sign(𝐠ₜ[i_t]) # counterfactual update
        P[i_t] += 1 # update 
        t += 1 # update number of times feature is changed
        converged = posterior_predictive(classifier, x̲)[1] .> Γ # check if converged
        max_number_changes_reached = all(P.==n)
        path = vcat(path, reshape(x̲, 1, D))
    end
    
    # Output:
    y̲ = predict(classifier, x̲; proba=false)[1]
    valid = y̲ == target * 1.0
    cost = norm(x.-x̲)
    if (constant_needed)
        path = path[:,2:end]
        x̲ = x̲[2:end]
        x = x[2:end]
    end
    recourse = Recourse(x̲, y̲, path, target, valid, cost, x) 
    
    return recourse
end;