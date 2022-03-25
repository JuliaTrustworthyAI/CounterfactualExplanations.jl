################################################################################
# --------------- Base type for model:
################################################################################
"""
AbstractFittedModel

Base type for fitted models.
"""
abstract type AbstractFittedModel end

# -------- Linear Logistic Model:
# This is an example of how to construct a AbstractFittedModel subtype:
"""
    LogisticModel(W::Matrix,b::AbstractArray)

Constructs a logistic classifier based on arrays containing coefficients `w` and constant terms `b`.

# Examples

```julia-repl
w = [1.0 -2.0] # estimated coefficients
b = [0] # estimated constant
𝑴 = CounterfactualExplanations.Models.LogisticModel(w, b);
```

See also: 
- [`logits(𝑴::LogisticModel, X::AbstractArray)`](@ref)
- [`probs(𝑴::LogisticModel, X::AbstractArray)`](@ref)
"""
struct LogisticModel <: AbstractFittedModel
    W::Matrix
    b::AbstractArray
end

# What follows are the two required outer methods:
"""
    logits(𝑴::LogisticModel, X::AbstractArray)

Computes logits as `WX+b`.

# Examples

```julia-repl
using CounterfactualExplanations.Models
w = [1.0 -2.0] # estimated coefficients
b = [0] # estimated constant
𝑴 = LogisticModel(w, b);
x = [1,1]
logits(𝑴, x)
```

See also [`LogisticModel(W::Matrix,b::AbstractArray)`](@ref).
"""
logits(𝑴::LogisticModel, X::AbstractArray) = 𝑴.W*X .+ 𝑴.b

"""
    probs(𝑴::LogisticModel, X::AbstractArray)

Computes predictive probabilities from logits as `σ(WX+b)` where 'σ' is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function). 

# Examples

```julia-repl
using CounterfactualExplanations.Models
w = [1.0 -2.0] # estimated coefficients
b = [0] # estimated constant
𝑴 = LogisticModel(w, b);
x = [1,1]
probs(𝑴, x)
```

See also [`LogisticModel(W::Matrix,b::AbstractArray)`](@ref).
"""
probs(𝑴::LogisticModel, X::AbstractArray) = NNlib.σ.(logits(𝑴, X))

# -------- Bayesian model:
"""
    BayesianLogisticModel(μ::Matrix,Σ::Matrix)

Constructs a Bayesian logistic classifier based on maximum a posteriori (MAP) estimates `μ` (coefficients including constant term(s)) and `Σ` (covariance matrix). 

# Examples

```julia-repl
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0 1.0 -2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
𝑴 = CounterfactualExplanations.Models.BayesianLogisticModel(μ, Σ);
```

See also:
- [`logits(𝑴::BayesianLogisticModel, X::AbstractArray)`](@ref)
- [`probs(𝑴::BayesianLogisticModel, X::AbstractArray)`](@ref)
"""
struct BayesianLogisticModel <: AbstractFittedModel
    μ::Matrix
    Σ::Matrix
    BayesianLogisticModel(μ, Σ) = length(μ)^2 != length(Σ) ? throw(DimensionMismatch("Dimensions of μ and its covariance matrix Σ do not match.")) : new(μ, Σ)
end

# What follows are the three required outer methods:
"""
    logits(𝑴::BayesianLogisticModel, X::AbstractArray)

Computes logits as `μ[1ᵀ Xᵀ]ᵀ`.

# Examples

```julia-repl
using CounterfactualExplanations.Models
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0 1.0 -2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
𝑴 = BayesianLogisticModel(μ, Σ);
x = [1,1]
logits(𝑴, x)
```

See also [`BayesianLogisticModel(μ::Matrix,Σ::Matrix)`](@ref)
"""
function logits(𝑴::BayesianLogisticModel, X::AbstractArray)
    if !isa(X, AbstractMatrix)
        X = reshape(X, length(X), 1)
    end
    X = vcat(ones(size(X)[2])', X) # add for constant
    return 𝑴.μ * X
end

"""
    probs(𝑴::BayesianLogisticModel, X::AbstractArray)

Computes predictive probabilities using a Probit approximation. 

# Examples

```julia-repl
using CounterfactualExplanations.Models
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0 1.0 -2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
𝑴 = BayesianLogisticModel(μ, Σ);
x = [1,1]
probs(𝑴, x)
```

See also [`BayesianLogisticModel(μ::Matrix,Σ::Matrix)`](@ref)
"""
function probs(𝑴::BayesianLogisticModel, X::AbstractArray)
    μ = 𝑴.μ # MAP mean vector
    Σ = 𝑴.Σ # MAP covariance matrix
    # Inner product:
    z = logits(𝑴, X)
    # Probit approximation
    if !isa(X, AbstractMatrix)
        X = reshape(X, length(X), 1)
    end
    X = vcat(ones(size(X)[2])', X) # add for constant
    v = [X[:,n]'Σ*X[:,n] for n=1:size(X)[2]]    
    κ = 1 ./ sqrt.(1 .+ π/8 .* v) # scaling factor for logits
    z = κ' .* z
    # Compute probabilities
    p = NNlib.σ.(z)
    p = size(p)[2] == 1 ? vec(p) : p
    return p
end