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
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random

# load data
function get_data(X::AbstractArray, y::AbstractArray, batch_size)
    DataLoader((X, y), batchsize=batch_size, shuffle=true)
end

struct Encoder
    linear
    μ
    logσ
end
@functor Encoder
    
Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
    Dense(input_dim, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # logσ
)

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

function rand(encoder::Encoder, x, device=cpu)
    μ, logσ = encoder(x)
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    return z, μ, logσ
end


Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
)

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
    loss = model_loss
    opt = ADAM(η)
end

# VAE:
struct VAE <: AbstractGenerativeModel
    encoder::Encoder
    decoder::Any
    params::VAEParams
end

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

function reconstruct(generative_model::VAE, x, device=cpu)
    z, μ, logσ = rand(generative_model.encoder, x, device)
    generative_model.decoder(z), μ, logσ
end

function model_loss(generative_model::VAE, λ, x, device)
    z, μ, logσ = reconstruct(generative_model, x, device)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len

    logp_x_z = -logitbinarycrossentropy(z, x, agg=sum) / len
    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(generative_model.decoder))
    
    -logp_x_z + kl_q_p + reg
end

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

        for (x, _) in loader 
            
            loss, back = Flux.pullback(ps) do
                model_loss(generative_model, args.λ, x |> args.device, args.device)
            end
            @info "Loss: $loss"
            grad = back(1f0)
            Flux.Optimise.update!(args.opt, ps, grad)

            # progress meter
            next!(progress; showvalues=[(:loss, loss)]) 

            train_steps += 1
        end 
    end
end
