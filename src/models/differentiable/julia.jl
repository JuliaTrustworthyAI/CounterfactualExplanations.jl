#############################################################################
# Flux models
#############################################################################
abstract type AbstractDifferentiableJuliaModel <: AbstractDifferentiableModel end

using Flux
# Constructor
struct FluxModel <: Models.AbstractDifferentiableJuliaModel
    model::Any
    likelihood::Symbol
    function FluxModel(model, likelihood)
        if likelihood ∈ [:classification_binary,:classification_multi]
            new(model, likelihood)
        else
            throw(ArgumentError("`type` should be in `[:classification_binary,:classification_multi]`"))
        end
    end
end

# Outer constructor method:
function FluxModel(model; likelihood::Symbol=:classification_binary)
    FluxModel(model, likelihood)
end

# Methods
function logits(M::FluxModel, X::AbstractArray)
    return M.model(X)
end

function probs(M::FluxModel, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = σ.(logits(M, X))
    elseif M.likelihood == :classification_multi
        output = softmax(logits(M, X))
    end
    return output
end

#############################################################################
# Laplace Redux
#############################################################################
using Flux, LaplaceRedux
# Constructor
struct LaplaceReduxModel <: Models.AbstractDifferentiableJuliaModel
    model::Laplace
    likelihood::Symbol
    function LaplaceReduxModel(model, likelihood)
        if likelihood == :classification_binary
            new(model, likelihood)
        elseif likelihood==:classification_multi
            throw(ArgumentError("`type` should be `:classification_binary`. Support for multi-class Laplace Redux is not yet implemented."))
        else
            throw(ArgumentError("`type` should be in `[:classification_binary,:classification_multi]`"))
        end
    end
end

# Outer constructor method:
function LaplaceReduxModel(model; likelihood::Symbol=:classification_binary)
    LaplaceReduxModel(model, likelihood)
end

# Methods
logits(M::LaplaceReduxModel, X::AbstractArray) = M.model.model(X)
probs(M::LaplaceReduxModel, X::AbstractArray)= LaplaceRedux.predict(M.model, X)


#############################################################################
# Gradient Boosted Trees
#############################################################################
using EvoTrees

struct EvoTreeModel <: Models.AbstractDifferentiableJuliaModel
    model::EvoTrees.GBTree
    likelihood::Symbol
    function EvoTreeModel(model, likelihood)
        if likelihood ∈ [:classification_binary,:classification_multi]
            new(model, likelihood)
        else
            throw(ArgumentError("`type` should be in `[:classification_binary,:classification_multi]`"))
        end
    end
end

# Outer constructor method:
function EvoTreeModel(model; likelihood::Symbol=:classification_binary)
    EvoTreeModel(model, likelihood)
end

# Methods
function logits(M::EvoTreeModel, X::AbstractArray)
    p = probs(M, X)
    if M.likelihood == :classification_binary
        output = log.(p./(1 .- p))
    else
        output = log.(p)
    end
    return output
end

function probs(M::EvoTreeModel, X::AbstractArray{<:Number, 2})
    output = EvoTrees.predict(M.model, X')'
    if M.likelihood == :classification_binary
        output = reshape(output[2,:],1,size(output,2)) # binary case
    end
    return output
end

function probs(M::EvoTreeModel, X::AbstractArray{<:Number, 1})
    X = reshape(X, 1,length(X))
    output = EvoTrees.predict(M.model, X)'
    if M.likelihood == :classification_binary
        output = reshape(output[2,:],1,size(output,2)) # binary case
    end
    return output
end

function probs(M::EvoTreeModel, X::AbstractArray{<:Number, 3})
    output = mapslices(x -> probs(M,x), X, dims=[1,2])
    return output
end


#############################################################################
# Baseline classifiers for illustrative purposes 
#############################################################################

# -------- Linear Logistic Model:
"""
    LogisticModel(W::Matrix,b::AbstractArray)

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
struct LogisticModel <: Models.AbstractDifferentiableJuliaModel
    W::Matrix
    b::AbstractArray
    likelihood::Symbol
end

LogisticModel(W,b;likelihood=:classification_binary) = LogisticModel(W,b,likelihood)

using MLUtils
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

See also [`LogisticModel(W::Matrix,b::AbstractArray)`](@ref).
"""
function logits(M::LogisticModel, X::AbstractArray)
    if ndims(X) == 3
        n = size(X,3)
        reshape(map(x -> (M.W*x .+ M.b)[1], [X[:,:,i] for i ∈ 1:n]),1,1,n)
        # mapslices(x -> M.W*x .+ M.b, X, dims=(1,2)) 
    else
        M.W*X .+ M.b
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

See also [`LogisticModel(W::Matrix,b::AbstractArray)`](@ref).
"""
probs(M::LogisticModel, X::AbstractArray) = Flux.σ.(logits(M, X))

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
M = CounterfactualExplanations.Models.BayesianLogisticModel(μ, Σ);
```

See also:
- [`logits(M::BayesianLogisticModel, X::AbstractArray)`](@ref)
- [`probs(M::BayesianLogisticModel, X::AbstractArray)`](@ref)
"""
struct BayesianLogisticModel <: Models.AbstractDifferentiableJuliaModel
    μ::Matrix
    Σ::Matrix
    likelihood::Symbol
    BayesianLogisticModel(μ, Σ, likelihood) = length(μ)^2 != length(Σ) ? throw(DimensionMismatch("Dimensions of μ and its covariance matrix Σ do not match.")) : new(μ, Σ, likelihood)
end

BayesianLogisticModel(μ,Σ;likelihood=:classification_binary) = BayesianLogisticModel(μ,Σ,likelihood)

# What follows are the three required outer methods:
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

See also [`BayesianLogisticModel(μ::Matrix,Σ::Matrix)`](@ref)
"""
function logits(M::BayesianLogisticModel, X::AbstractArray)
    if !isa(X, AbstractMatrix)
        X = reshape(X, length(X), 1)
    end
    X = vcat(ones(size(X)[2])', X) # add for constant
    return M.μ * X
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

See also [`BayesianLogisticModel(μ::Matrix,Σ::Matrix)`](@ref)
"""
function probs(M::BayesianLogisticModel, X::AbstractArray)
    μ = M.μ # MAP mean vector
    Σ = M.Σ # MAP covariance matrix
    # Inner product:
    z = logits(M, X)
    # Probit approximation
    if !isa(X, AbstractMatrix)
        X = reshape(X, length(X), 1)
    end
    X = vcat(ones(size(X)[2])', X) # add for constant
    v = [X[:,n]'Σ*X[:,n] for n=1:size(X)[2]]    
    κ = 1 ./ sqrt.(1 .+ π/8 .* v) # scaling factor for logits
    z = κ' .* z
    # Compute probabilities
    p = Flux.σ.(z)
    p = size(p)[2] == 1 ? vec(p) : p
    return p
end



