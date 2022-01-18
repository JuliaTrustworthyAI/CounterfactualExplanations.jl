# --------------- Base type for model:
module Models

using Flux, LinearAlgebra

abstract type FittedModel end


# -------- Linear Logistic Model:
# This is an example of how to construct a FittedModel subtype:
"""
    LogisticModel(w::AbstractArray,b::AbstractArray)

Constructs a logistic classifier based on arrays containing coefficients `w` and constant terms `b`.

# Examples

```julia-repl
w = [1.0,-2.0] # estimated coefficients
b = [0] # estimated constant
𝑴 = AlgorithmicRecourse.Models.LogisticModel(w, b);
```

See also: 
- [`logits(𝑴::LogisticModel, X::AbstractArray)`](@ref)
- [`probs(𝑴::LogisticModel, X::AbstractArray)`](@ref)
"""
struct LogisticModel <: FittedModel
    w::AbstractArray
    b::AbstractArray
end

# What follows are the two required outer methods:
"""
    logits(𝑴::LogisticModel, X::AbstractArray)

Computes logits as `Xw+b`.

# Examples

```julia-repl
w = [1.0,-2.0] # estimated coefficients
b = [0] # estimated constant
𝑴 = AlgorithmicRecourse.Models.LogisticModel(w, b);
x = reshape([1,1],1,2)
logits(𝑴, x)
```

See also [`LogisticModel(w::AbstractArray,b::AbstractArray)`](@ref).
"""
logits(𝑴::LogisticModel, X::AbstractArray) = X * 𝑴.w .+ 𝑴.b

"""
    probs(𝑴::LogisticModel, X::AbstractArray)

Computes predictive probabilities from logits as `σ(Xw+b)` where 'σ' is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function). 

# Examples

```julia-repl
w = [1.0,-2.0] # estimated coefficients
b = [0] # estimated constant
𝑴 = AlgorithmicRecourse.Models.LogisticModel(w, b);
x = reshape([1,1],1,2)
probs(𝑴, x)
```

See also [`LogisticModel(w::AbstractArray,b::AbstractArray)`](@ref).
"""
probs(𝑴::LogisticModel, X::AbstractArray) = Flux.σ.(logits(𝑴, X))

# -------- Bayesian model:
"""
    BayesianLogisticModel(w::AbstractArray,b::AbstractArray)

Constructs a Bayesian logistic classifier based on maximum a posteriori (MAP) estimates `μ` (coefficients including constant term(s)) and `Σ` (covariance matrix). 

# Examples

```julia-repl
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0, 1.0,-2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
𝑴 = AlgorithmicRecourse.Models.BayesianLogisticModel(μ, Σ);
```

See also:
- [`logits(𝑴::BayesianLogisticModel, X::AbstractArray)`](@ref)
- [`probs(𝑴::BayesianLogisticModel, X::AbstractArray)`](@ref)
"""
struct BayesianLogisticModel <: FittedModel
    μ::AbstractArray
    Σ::AbstractArray
end

# What follows are the three required outer methods:
"""
    logits(𝑴::BayesianLogisticModel, X::AbstractArray)

Computes logits as `[1 X]μ`.

# Examples

```julia-repl
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0, 1.0,-2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
𝑴 = AlgorithmicRecourse.Models.BayesianLogisticModel(μ, Σ);
x = reshape([1,1],1,2)
logits(𝑴, x)
```

See also [`BayesianLogisticModel(w::AbstractArray,b::AbstractArray)`](@ref)
"""
logits(𝑴::BayesianLogisticModel, X::AbstractArray) = hcat(1, X) * 𝑴.μ

"""
    probs(𝑴::BayesianLogisticModel, X::AbstractArray)

Computes predictive probabilities using a Probit approximation. 

# Examples

```julia-repl
using Random, LinearAlgebra
Random.seed!(1234)
μ = [0, 1.0,-2.0] # MAP coefficients
Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
𝑴 = AlgorithmicRecourse.Models.BayesianLogisticModel(μ, Σ);
x = reshape([1,1],1,2)
probs(𝑴, x)
```

See also [`BayesianLogisticModel(w::AbstractArray,b::AbstractArray)`](@ref)
"""
function probs(𝑴::BayesianLogisticModel, X::AbstractArray)
    μ = 𝑴.μ # MAP mean vector
    Σ = 𝑴.Σ # MAP covariance matrix
    if !isa(X, Matrix)
        X = reshape(X, 1, length(X))
    end
    X = hcat(ones(size(X)[1]), X) # add for constant
    # Inner product:
    z = X*μ
    # Probit approximation
    v = [X[n,:]'Σ*X[n,:] for n=1:size(X)[1]]
    κ = 1 ./ sqrt.(1 .+ π/8 .* v) 
    z = κ .* z
    # Compute probabilities
    p = Flux.σ.(z)
    return p
end

end