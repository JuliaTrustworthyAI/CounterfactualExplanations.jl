using Flux: Flux, Adam, cpu, gpu
using ProgressMeter: Progress, next!
using Statistics: mean

"""
    VAEParams <: AbstractGMParams

The default VAE parameters describing both the encoder/decoder architecture and the training process.
"""
Base.@kwdef mutable struct VAEParams <: AbstractGMParams
    η = 1e-3                # learning rate
    λ = 0.01f0              # regularization parameter
    batch_size = 50         # batch size
    epochs = 100            # number of epochs
    seed = 0                # random seed
    gpu = true             # use GPU
    device = gpu            # default device
    latent_dim = 2          # latent dimension
    hidden_dim = 32         # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    nll = Flux.Losses.mse               # negative log likelihood -log(p(x|z)): MSE for Gaussian, logit binary cross-entropy for Bernoulli
    opt = Adam(η)           # optimizer
end

"""
    VAE <: AbstractGenerativeModel

Constructs the Variational Autoencoder. The VAE is a subtype of `AbstractGenerativeModel`. Any (sub-)type of `AbstractGenerativeModel` is accepted by latent space generators. 
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

    # load hyperparameters
    args = VAEParams(; kws...)

    # GPU config
    if args.gpu
        args.device = gpu
    else
        args.device = cpu
    end

    # initialize encoder and decoder
    encoder = args.device(Encoder(input_dim, args.latent_dim, args.hidden_dim))
    decoder = args.device(Decoder(input_dim, args.latent_dim, args.hidden_dim))

    return VAE(encoder, decoder, args, false)
end

Flux.@layer VAE

function Flux.trainable(generative_model::VAE)
    return (encoder=generative_model.encoder, decoder=generative_model.decoder)
end

"""
    reconstruct(generative_model::VAE, x, device=cpu)

Implements a full pass of some input `x` through the VAE: `x ↦ x̂`.
"""
function reconstruct(generative_model::VAE, x, device=cpu)
    z, μ, logσ = rand(generative_model.encoder, x, device)
    return generative_model.decoder(z), μ, logσ
end

function model_loss(generative_model::VAE, λ, x, device)
    z, μ, logσ = reconstruct(generative_model, x, device)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2.0f0 * logσ) + μ^2 - 1.0f0 - 2.0f0 * logσ)) / len
    # Negative log-likelihood: - log(p(x|z))
    nll_x_z = -generative_model.params.nll(z, x; agg=sum) / len
    # Weight regularization:
    reg = λ * sum(x -> sum(x .^ 2), Flux.params(generative_model.decoder))

    elbo = -nll_x_z + kl_q_p + reg

    return elbo
end

function _fit(generative_model::Type{VAE}, X::AbstractArray; kws...)
    generative_model = VAE(size(X, 1); kws...)

    # load hyperparamters
    args = generative_model.params
    args.seed > 0 && Random.seed!(args.seed)

    # load data
    loader = get_data(X, args.batch_size)

    # parameters
    ps = Flux.params(generative_model)

    # Verbosity
    if flux_training_params.verbose
        @info "Begin training VAE"
        p_epoch = Progress(
            args.epochs; desc="Progress on epochs:", showspeed=true, color=:green
        )
    end

    # training
    for epoch in 1:(args.epochs)
        avg_loss = []
        for (x,) in loader
            loss, back = Flux.pullback(ps) do
                model_loss(generative_model, args.λ, args.device(x), args.device)
            end

            avg_loss = vcat(avg_loss, loss)
            grad = back(1.0f0)
            Flux.Optimise.update!(args.opt, ps, grad)
        end

        avg_loss = mean(avg_loss)
        if flux_training_params.verbose
            next!(p_epoch; showvalues=[(:Loss, "$(avg_loss)")])
        end
    end

    # Set training status to true:
    generative_model.trained = true

    return generative_model
end

function train!(generative_model::VAE, X::AbstractArray; kws...)

    # load hyperparamters
    args = generative_model.params
    args.seed > 0 && Random.seed!(args.seed)

    # load data
    loader = get_data(X, args.batch_size)

    # parameters
    ps = Flux.params(generative_model)

    # Verbosity
    if flux_training_params.verbose
        @info "Begin training VAE"
        p_epoch = Progress(
            args.epochs; desc="Progress on epochs:", showspeed=true, color=:green
        )
    end

    # training
    for epoch in 1:(args.epochs)
        avg_loss = []
        for (x,) in loader
            loss, back = Flux.pullback(ps) do
                model_loss(generative_model, args.λ, args.device(x), args.device)
            end

            avg_loss = vcat(avg_loss, loss)
            grad = back(1.0f0)
            Flux.Optimise.update!(args.opt, ps, grad)
        end

        avg_loss = mean(avg_loss)
        if flux_training_params.verbose
            next!(p_epoch; showvalues=[(:Loss, "$(avg_loss)")])
        end
    end

    # Set training status to true:
    return generative_model.trained = true
end

function retrain!(generative_model::VAE, X::AbstractArray; n_epochs=10)

    # load hyperparameters
    args = generative_model.params
    args.seed > 0 && Random.seed!(args.seed)

    # load data
    loader = get_data(X, args.batch_size)

    # parameters
    ps = Flux.params(generative_model)

    # Verbosity
    if flux_training_params.verbose
        @info "Begin training VAE"
        p_epoch = Progress(
            args.epochs; desc="Progress on epochs:", showspeed=true, color=:green
        )
    end

    # training
    train_steps = 0
    for epoch in 1:n_epochs
        avg_loss = []
        for (x,) in loader
            loss, back = Flux.pullback(ps) do
                model_loss(generative_model, args.λ, args.device(x), args.device)
            end

            avg_loss = vcat(avg_loss, loss)
            grad = back(1.0f0)
            Flux.Optimise.update!(args.opt, ps, grad)

            train_steps += 1
        end

        avg_loss = mean(avg_loss)
        if flux_training_params.verbose
            next!(p_epoch; showvalues=[(:Loss, "$(avg_loss)")])
        end
    end
end

"""
    get_data(X::AbstractArray, batch_size)

Preparing data for mini-batch training .
"""
function get_data(X::AbstractArray, batch_size)
    return Flux.DataLoader((X,); batchsize=batch_size, shuffle=true)
end

"""
    get_data(X::AbstractArray, y::AbstractArray, batch_size)

Preparing data for mini-batch training .
"""
function get_data(X::AbstractArray, y::AbstractArray, batch_size)
    return Flux.DataLoader((X, y); batchsize=batch_size, shuffle=true)
end

"""
    encode(generative_model::VAE, x::AbstractArray)

Encodes an array `x` using the VAE encoder. Specifically, it samples from the latent distribution. It does so by first passing `x` through the encoder to obtain the mean and log-variance of the latent distribution. Then, it samples from the latent distribution using the reparameterization trick. See [`Random.rand(encoder::Encoder, x, device=cpu)`](@ref) for more details.
"""
function encode(generative_model::VAE, x::AbstractArray)
    x, _, _ = GenerativeModels.rand(generative_model.encoder, x)
    return x
end

"""
    decode(generative_model::VAE, x::AbstractArray)

Decodes an array `x` using the VAE decoder.
"""
function decode(generative_model::VAE, x::AbstractArray)
    if generative_model.params.nll == Flux.Losses.logitbinarycrossentropy
        x = Flux.σ.(generative_model.decoder(x))
    else
        x = generative_model.decoder(x)
    end
    return x
end
