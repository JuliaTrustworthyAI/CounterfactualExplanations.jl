
abstract type Objective end
struct GenericObjective <: Objective
    ℓ::Symbol
    cost::Symbol
end

function generic(x̅::Vector{x}, 𝓜, target::Float64, ℓ::Function, cost::Function)
    return ℓ(x̅::Vector{x}, 𝓜, target::Float64) + λ .* cost()
end
