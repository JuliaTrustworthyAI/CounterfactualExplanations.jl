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

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
)

# VAE:
struct VAE{enc<:Encoder,dec<:Any} <: AbstractGenerativeModel
    encoder::enc
    decoder::dec
end

VAE(enc::Encoder, dec::Any) = VAE(enc, dec)

Flux.@functor VAE

function Flux.trainable(m::VAE)
    (encoder=m.encoder, decoder=m.decoder)
end

function reconstuct(m::VAE, x, device)
    μ, logσ = m.encoder(x)
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    μ, logσ, m.decoder(z)
end

function model_loss(m::VAE, λ, x, device)
    μ, logσ, decoder_z = reconstuct(m, x, device)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len

    logp_x_z = -logitbinarycrossentropy(decoder_z, x, agg=sum) / len
    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(m.decoder))
    
    -logp_x_z + kl_q_p + reg
end

# arguments for the `train` function 
@with_kw mutable struct VAEParams
    η = 1e-3                # learning rate
    λ = 0.01f0              # regularization paramater
    batch_size = 128        # batch size
    sample_size = 10        # sampling size for output    
    epochs = 20             # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    latent_dim = 2          # latent dimension
    hidden_dim = 500        # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    loss = model_loss
    opt = ADAM(η)
end

function train(m::VAE, X::AbstractArray, y::AbstractArray; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load data
    loader = get_data(X, y, args.batch_size)
    
    # initialize encoder and decoder
    encoder = m.enc(args.input_dim, args.latent_dim, args.hidden_dim) |> device
    decoder = m.dec(args.input_dim, args.latent_dim, args.hidden_dim) |> device

    # ADAM optimizer
    opt = ADAM(args.η)
    
    # parameters
    ps = Flux.params(model)

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (x, _) in loader 
            
            loss, back = Flux.pullback(ps) do
                model_loss(m, args.λ, x |> device, device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # progress meter
            next!(progress; showvalues=[(:loss, loss)]) 

            train_steps += 1
        end 
    end
end
