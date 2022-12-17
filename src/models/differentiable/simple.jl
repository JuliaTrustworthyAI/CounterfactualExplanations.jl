using MLUtils
using SliceMap

"""
    LogisticModel <: AbstractDifferentiableJuliaModel

Constructs a logistic classifier based on arrays containing coefficients `w` and constant terms `b`.

# Examples

```julia-repl
w = [1.0 -2.0] # estimated coefficients
b = [0] # estimated constant
M = CounterfactualExplanations.Models.LogisticModel(w, b);
```

See also: 
- [`logits(M::LogisticModel, X::AbstractArray)`](@ref)
- [`probs(M::LogisticModel, X::AbstractArray)`](@ref)
"""
struct LogisticModel <: AbstractDifferentiableJuliaModel
    W::Matrix
    b::AbstractArray
    likelihood::Symbol
end

LogisticModel(W, b; likelihood = :classification_binary) = LogisticModel(W, b, likelihood)

# What follows are the two required outer methods:
"""
    logits(M::LogisticModel, X::AbstractArray)

Computes logits as `WX+b`.

# Examples

```julia-repl
using CounterfactualExplanations.Models
w = [1.0 -2.0] # estimated coefficients
b = [0] # estimated constant
M = LogisticModel(w, b);
x = [1,1]
logits(M, x)
```

See also [`LogisticModel <: AbstractDifferentiableJuliaModel`](@ref).
"""
function logits(M::LogisticModel, X::AbstractArray)
    if ndims(X) == 3
        n = size(X, 3)
        reshape(map(x -> (M.W*x.+M.b)[1], [X[:, :, i] for i ∈ 1:n]), 1, 1, n)
        # SliceMap.slicemap(x -> M.W*x .+ M.b, X, dims=(1,2)) 
    else
        M.W * X .+ M.b
    end
end

"""
    probs(M::LogisticModel, X::AbstractArray)

Computes predictive probabilities from logits as `σ(WX+b)` where 'σ' is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function). 

# Examples

```julia-repl
using CounterfactualExplanations.Models
w = [1.0 -2.0] # estimated coefficients
b = [0] # estimated constant
M = LogisticModel(w, b);
x = [1,1]
probs(M, x)
```

See also [`LogisticModel <: AbstractDifferentiableJuliaModel`](@ref).
"""
probs(M::LogisticModel, X::AbstractArray) = Flux.σ.(logits(M, X))


"""
    BayesianLogisticModel <: AbstractDifferentiableJuliaModel

Constructs a Bayesian logistic classifier based on maximum a posteriori (MAP) estimates `μ` (coefficients including constant term(s)) and `Σ` (covariance matrix). 

# Examples

```julia-repl
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0 1.0 -2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
M = CounterfactualExplanations.Models.BayesianLogisticModel(μ, Σ);
```

See also:
- [`logits(M::BayesianLogisticModel, X::AbstractArray)`](@ref)
- [`probs(M::BayesianLogisticModel, X::AbstractArray)`](@ref)
"""
struct BayesianLogisticModel <: AbstractDifferentiableJuliaModel
    μ::Matrix
    Σ::Matrix
    likelihood::Symbol
    BayesianLogisticModel(μ, Σ, likelihood) =
        length(μ)^2 != length(Σ) ?
        throw(
            DimensionMismatch("Dimensions of μ and its covariance matrix Σ do not match."),
        ) : new(μ, Σ, likelihood)
end

BayesianLogisticModel(μ, Σ; likelihood = :classification_binary) =
    BayesianLogisticModel(μ, Σ, likelihood)

"""
    logits(M::BayesianLogisticModel, X::AbstractArray)

Computes logits as `μ[1ᵀ Xᵀ]ᵀ`.

# Examples

```julia-repl
using CounterfactualExplanations.Models
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0 1.0 -2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
M = BayesianLogisticModel(μ, Σ);
x = [1,1]
logits(M, x)
```

See also [`BayesianLogisticModel <: AbstractDifferentiableJuliaModel`](@ref)
"""
function logits(M::BayesianLogisticModel, X::AbstractArray)

    X = vcat(1.0, X) # add for constant

    if ndims(X) == 3
        y = SliceMap.slicemap(x -> M.μ * x, X, dims = (1, 2))
    else
        y = M.μ * X
    end

    return y
end

"""
    probs(M::BayesianLogisticModel, X::AbstractArray)

Computes predictive probabilities using a Probit approximation. 

# Examples

```julia-repl
using CounterfactualExplanations.Models
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0 1.0 -2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
M = BayesianLogisticModel(μ, Σ);
x = [1,1]
probs(M, x)
```

See also [`BBayesianLogisticModel <: AbstractDifferentiableJuliaModel`](@ref)
"""
function probs(M::BayesianLogisticModel, X::AbstractArray)

    μ = M.μ             # MAP mean vector
    Σ = M.Σ             # MAP covariance matrix

    # Probit approximation
    function probit_approx(x)
        z = logits(M, x)
        x = vcat(1.0, x) # add for constant
        v = [_x'Σ * _x for _x in eachcol(x)]
        κ = 1 ./ sqrt.(1 .+ π / 8 .* v) # scaling factor for logits
        z = κ' .* z
        # Compute probabilities
        p = Flux.σ.(z)
        return p
    end

    if ndims(X) == 3
        p = SliceMap.slicemap(x -> probit_approx(x), X, dims = (1, 2))
    else
        p = probit_approx(X)
    end

    return p

end

# μ = M.μ                             # MAP mean vector
# Σ = M.Σ                             # MAP covariance matrix
# z = logits(M, X)                    # logits

# # Probit approximation
# X = vcat(1, X)                      # add for constant
# v = collect(SliceMap.slicemap(x -> x'Σ * x, X, dims=(1,2)))
# κ = 1 ./ sqrt.(1 .+ π/8 .* v)       # scaling factor for logits
# println("kappa:$(size(κ))")
# z = collect(SliceMap.slicemap(x -> x' * z, κ, dims=(1,2)))
# # Compute probabilities
# p = collect(SliceMap.slicemap(x -> Flux.σ.(x), z, dims=(1,2)))
# return p
