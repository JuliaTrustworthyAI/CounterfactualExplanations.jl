using Flux: sigmoid
using Random: Random

"""
Encoder

Constructs encoder part of VAE: a simple Flux neural network with one hidden layer and two linear output layers for the first two moments of the latent distribution.
"""
struct Encoder
    linear::Any
    μ::Any
    logσ::Any
end
Flux.@layer Encoder

function Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation=sigmoid)
    return Encoder(
        Flux.Dense(input_dim, hidden_dim, activation),       # linear
        Flux.Dense(hidden_dim, latent_dim),                  # μ
        Flux.Dense(hidden_dim, latent_dim),                  # logσ
    )
end

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    return encoder.μ(h), encoder.logσ(h)
end

"""
    Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation=relu)

The default decoder architecture is just a Flux Chain with one hidden layer and a linear output layer. 
"""
function Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation=tanh)
    return Flux.Chain(
        Flux.Dense(latent_dim, hidden_dim, activation), Flux.Dense(hidden_dim, input_dim)
    )
end

"""
reparameterization_trick(μ,logσ,device=cpu)

Helper function that implements the reparameterization trick: `z ∼ 𝒩(μ,σ²) ⇔ z=μ + σ ⊙ ε, ε ∼ 𝒩(0,I).`
"""
function reparameterization_trick(μ, logσ, device=cpu)
    return μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
end

"""
Random.rand(encoder::Encoder, x, device=cpu)

Draws random samples from the latent distribution.
"""
function Random.rand(encoder::Encoder, x, device=cpu)
    μ, logσ = encoder(x)
    z = reparameterization_trick(μ, logσ, device)
    return z, μ, logσ
end
