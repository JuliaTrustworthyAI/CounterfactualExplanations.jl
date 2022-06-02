# Variational Autoencoder(VAE)
#
# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling
# https://arxiv.org/abs/1312.6114

# Adopted from Flux Model zoo: 
# https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl

using BSON
using CUDA
using Flux
using Flux: @functor, chunk
using Flux.Losses: logitbinarycrossentropy, mse
using Flux.Data: DataLoader
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random

"""
    get_data(X::AbstractArray, y::AbstractArray, batch_size)

Preparing data for mini-batch training .
"""
function get_data(X::AbstractArray, y::AbstractArray, batch_size)
    DataLoader((X, y), batchsize=batch_size, shuffle=true)
end

"""
    Encoder

Constructs encoder part of VAE: a simple Flux neural network with one hidden layer and two linear output layers for the first two moments of the latent distribution.
"""
struct Encoder
    linear
    μ
    logσ
end
@functor Encoder
    
Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation=relu) = Encoder(
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
function reparameterization_trick(μ,logσ,device=cpu)
    return μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
end

import Random: rand
"""
    rand(encoder::Encoder, x, device=cpu)

Draws random samples from the latent distribution.
"""
function rand(encoder::Encoder, x, device=cpu)
    μ, logσ = encoder(x)
    z = reparameterization_trick(μ, logσ)
    return z, μ, logσ
end

"""
    Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation=relu)

The default decoder architecture is just a Flux Chain with one hidden layer and a linear output layer. 
"""
Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int; activation=relu) = Chain(
    Dense(latent_dim, hidden_dim, activation),
    Dense(hidden_dim, input_dim)
)

"""
    VAEParams <: AbstractGMParams

The default VAE parameters describing both the encoder/decoder architecture and the training process.
"""
@with_kw mutable struct VAEParams <: AbstractGMParams
    η = 1e-3                # learning rate
    λ = 0.01f0              # regularization paramater
    batch_size = 128        # batch size
    sample_size = 10        # sampling size for output    
    epochs = 20             # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    device = gpu            # default device
    latent_dim = 2          # latent dimension
    hidden_dim = 500        # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    nll = mse               # negative log likelihood -log(p(x|z)): MSE for Gaussian, logit binary cross-entropy for Bernoulli
    opt = ADAM(η)           # optimizer
end

"""
    VAE <: AbstractGenerativeModel

Constructrs the Variational Autoencoder. The VAE is a subtype of `AbstractGenerativeModel`. Any (sub-)type of `AbstractGenerativeModel` is accepted by latent space generators. 
"""
struct VAE <: AbstractGenerativeModel
    encoder::Encoder
    decoder::Any
    params::VAEParams
end

"""
    VAE(input_dim;kws...)

Outer method for instantiating a VAE.
"""
function VAE(input_dim;kws...)

    # load hyperparamters
    args = VAEParams(;kws...)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        args.device = gpu
        @info "Moving to GPU"
    else
        args.device = cpu
        @info "Moving to CPU"
    end

    # initialize encoder and decoder
    encoder = Encoder(input_dim, args.latent_dim, args.hidden_dim) |> args.device
    decoder = Decoder(input_dim, args.latent_dim, args.hidden_dim) |> args.device

    VAE(encoder, decoder, args)

end

Flux.@functor VAE

function Flux.trainable(generative_model::VAE)
    (encoder=generative_model.encoder, decoder=generative_model.decoder)
end

"""
    reconstruct(generative_model::VAE, x, device=cpu)

Implements a full pass of some input `x` through the VAE: `x ↦ x̂`.
"""
function reconstruct(generative_model::VAE, x, device=cpu)
    z, μ, logσ = rand(generative_model.encoder, x, device)
    generative_model.decoder(z), μ, logσ
end

"""
    


"""
function model_loss(generative_model::VAE, λ, x, device)
    
    z, μ, logσ = reconstruct(generative_model, x, device)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len
    # Negative log-likelihood: - log(p(x|z))
    nll_x_z = generative_model.params.nll(z, x, agg=sum) / len
    # Weight regularization:
    reg = λ * sum(x->sum(x.^2), Flux.params(generative_model.decoder))
    
    elbo = nll_x_z + kl_q_p + reg

    return elbo
end

using Statistics
function train!(generative_model::VAE, X::AbstractArray, y::AbstractArray; kws...)
    # load hyperparamters
    args = generative_model.params
    args.seed > 0 && Random.seed!(args.seed)

    # load data
    loader = get_data(X, y, args.batch_size)
    
    # parameters
    ps = Flux.params(generative_model)

    # training
    train_steps = 0
    @info "Start training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))
        avg_loss = []
        for (x, _) in loader 
            
            loss, back = Flux.pullback(ps) do
                model_loss(generative_model, args.λ, x |> args.device, args.device)
            end
            
            avg_loss = vcat(avg_loss, loss)
            grad = back(1f0)
            Flux.Optimise.update!(args.opt, ps, grad)

            # progress meter
            next!(progress; showvalues=[(:Loss, loss)]) 

            train_steps += 1
        end 
        avg_loss = mean(avg_loss)
        @info "Loss (avg): $avg_loss"
    end
end

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

using Plots
import Plots: plot
function plot(generative_model::VAE, X::AbstractArray, y::AbstractArray; image_data=false, kws...)
    args = generative_model.params
    device = args.device
    encoder, decoder = generative_model.encoder |> device, generative_model.decoder |> device
    
    # load data
    loader = get_data(X,y,args.batch_size)
    labels = []
    μ_1 = []
    μ_2 = []

    # clustering in the latent space
    # visualize first two dims
    out_dim = size(y)[1]
    pal = out_dim > 1 ? cgrad(:rainbow, out_dim, categorical = true) : :rainbow
    println(pal)
    plt_clustering = scatter(palette=pal, title="Latent space clustering")
    for (i, (x, y)) in enumerate(loader)
        i < 20 || break
        μ_i, _ = encoder(x |> device)
        μ_1 = vcat(μ_1, μ_i[1, :])
        μ_2 = vcat(μ_2, μ_i[2, :])
        
        y = out_dim == 1 ? y : Flux.onecold(y,1:(out_dim))
        labels = Int.(vcat(labels, vec(y)))
    end

    scatter!(
        μ_1, μ_2, 
        markerstrokewidth=0, markeralpha=0.8,
        aspect_ratio=1,
        markercolor=labels,
        group=labels
    )

    if image_data 
        z = range(-2.0, stop=2.0, length=11)
        len = Base.length(z)
        z1 = repeat(z, len)
        z2 = sort(z1)
        x = zeros(Float32, args.latent_dim, len^2) |> device
        x[1, :] = z1
        x[2, :] = z2
        samples = decoder(x)
        samples = sigmoid.(samples)
        samples = convert_to_image(samples, len)
        plt_manifold = Plots.plot(samples, axis=nothing, title="2D Manifold")
        plt = Plots.plot(plt_clustering, plt_manifold)
    else
        plt = plt_clustering
    end

    return plt
end
