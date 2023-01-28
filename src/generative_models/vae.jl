# Variational Autoencoder(VAE)
#
# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling
# https://arxiv.org/abs/1312.6114

# Adopted from Flux Model zoo: 
# https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl

using CUDA
using Flux
using Flux: @functor, chunk, DataLoader
using Flux.Losses: logitbinarycrossentropy, mse
using Parameters: @with_kw
using ProgressMeter
using Random
using Statistics

"""
    get_data(X::AbstractArray, y::AbstractArray, batch_size)

Preparing data for mini-batch training .
"""
function get_data(X::AbstractArray, y::AbstractArray, batch_size)
    DataLoader((X, y), batchsize = batch_size, shuffle = true)
end

"""
    Encoder

Constructs encoder part of VAE: a simple Flux neural network with one hidden layer and two linear output layers for the first two moments of the latent distribution.
"""
struct Encoder
    linear::Any
    μ::Any
    logσ::Any
end
@functor Encoder

Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation = relu) = Encoder(
    Dense(input_dim, hidden_dim, activation),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # logσ
)

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

"""
    reparameterization_trick(μ,logσ,device=cpu)

Helper function that implements the reparameterization trick: `z ∼ 𝒩(μ,σ²) ⇔ z=μ + σ ⊙ ε, ε ∼ 𝒩(0,I).`
"""
function reparameterization_trick(μ, logσ, device = cpu)
    return μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
end

"""
    Random.rand(encoder::Encoder, x, device=cpu)

Draws random samples from the latent distribution.
"""
function Random.rand(encoder::Encoder, x, device = cpu)
    μ, logσ = encoder(x)
    z = reparameterization_trick(μ, logσ)
    return z, μ, logσ
end

"""
    Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation=relu)

The default decoder architecture is just a Flux Chain with one hidden layer and a linear output layer. 
"""
Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation = relu) =
    Chain(Dense(latent_dim, hidden_dim, activation), Dense(hidden_dim, input_dim))

"""
    VAEParams <: AbstractGMParams

The default VAE parameters describing both the encoder/decoder architecture and the training process.
"""
@with_kw mutable struct VAEParams <: AbstractGMParams
    η = 1e-3                # learning rate
    λ = 0.01f0              # regularization paramater
    batch_size = 50         # batch size
    epochs = 100            # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    device = gpu            # default device
    latent_dim = 2          # latent dimension
    hidden_dim = 32         # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    nll = mse               # negative log likelihood -log(p(x|z)): MSE for Gaussian, logit binary cross-entropy for Bernoulli
    opt = Adam(η)           # optimizer
end

"""
    VAE <: AbstractGenerativeModel

Constructrs the Variational Autoencoder. The VAE is a subtype of `AbstractGenerativeModel`. Any (sub-)type of `AbstractGenerativeModel` is accepted by latent space generators. 
"""
mutable struct VAE <: AbstractGenerativeModel
    encoder::Encoder
    decoder::Any
    params::VAEParams
    trained::Bool
end

"""
    VAE(input_dim;kws...)

Outer method for instantiating a VAE.
"""
function VAE(input_dim; kws...)

    # load hyperparamters
    args = VAEParams(; kws...)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        args.device = gpu
    else
        args.device = cpu
    end

    # initialize encoder and decoder
    encoder = Encoder(input_dim, args.latent_dim, args.hidden_dim) |> args.device
    decoder = Decoder(input_dim, args.latent_dim, args.hidden_dim) |> args.device

    VAE(encoder, decoder, args, false)

end

Flux.@functor VAE

function Flux.trainable(generative_model::VAE)
    (encoder = generative_model.encoder, decoder = generative_model.decoder)
end

"""
    reconstruct(generative_model::VAE, x, device=cpu)

Implements a full pass of some input `x` through the VAE: `x ↦ x̂`.
"""
function reconstruct(generative_model::VAE, x, device = cpu)
    z, μ, logσ = rand(generative_model.encoder, x, device)
    generative_model.decoder(z), μ, logσ
end

"""
    


"""
function model_loss(generative_model::VAE, λ, x, device)

    z, μ, logσ = reconstruct(generative_model, x, device)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2.0f0 * logσ) + μ^2 - 1.0f0 - 2.0f0 * logσ)) / len
    # Negative log-likelihood: - log(p(x|z))
    nll_x_z = generative_model.params.nll(z, x, agg = sum) / len
    # Weight regularization:
    reg = λ * sum(x -> sum(x .^ 2), Flux.params(generative_model.decoder))

    elbo = nll_x_z + kl_q_p + reg

    return elbo
end

function train!(generative_model::VAE, X::AbstractArray, y::AbstractArray; kws...)

    # load hyperparamters
    args = generative_model.params
    args.seed > 0 && Random.seed!(args.seed)

    # load data
    loader = get_data(X, y, args.batch_size)

    # parameters
    ps = Flux.params(generative_model)

    # Verbosity
    if flux_training_params.verbose
        @info "Begin training VAE"
        p_epoch = Progress(args.epochs; desc="Progress on epochs:", showspeed=true, color=:green)
    end

    # training
    train_steps = 0
    for epoch = 1:args.epochs

        avg_loss = []
        for (x, _) in loader

            loss, back = Flux.pullback(ps) do
                model_loss(generative_model, args.λ, x |> args.device, args.device)
            end

            avg_loss = vcat(avg_loss, loss)
            grad = back(1.0f0)
            Flux.Optimise.update!(args.opt, ps, grad)

            train_steps += 1
        end

        avg_loss = mean(avg_loss)
        if flux_training_params.verbose
            next!(p_epoch, showvalues=[(:Loss, "$(avg_loss)")])
        end

    end

    # Set training status to true:
    generative_model.trained = true

end

function retrain!(generative_model::VAE, X::AbstractArray, y::AbstractArray; n_epochs = 10)

    # load hyperparamters
    args = generative_model.params
    args.seed > 0 && Random.seed!(args.seed)

    # load data
    loader = get_data(X, y, args.batch_size)

    # parameters
    ps = Flux.params(generative_model)

    # training
    train_steps = 0
    for epoch = 1:n_epochs

        avg_loss = []
        for (x, _) in loader

            loss, back = Flux.pullback(ps) do
                model_loss(generative_model, args.λ, x |> args.device, args.device)
            end

            avg_loss = vcat(avg_loss, loss)
            grad = back(1.0f0)
            Flux.Optimise.update!(args.opt, ps, grad)

            train_steps += 1
        end

        avg_loss = mean(avg_loss)
        if flux_training_params.verbose
            next!(p_epoch, showvalues=[(:Loss, "$(avg_loss)")])
        end

    end
end
