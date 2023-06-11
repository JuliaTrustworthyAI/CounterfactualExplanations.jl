"""
    VAEParams <: AbstractGMParams

The default VAE parameters describing both the encoder/decoder architecture and the training process.
"""
Parameters.@with_kw mutable struct VAEParams <: AbstractGMParams
    η = 1e-3                # learning rate
    λ = 0.01f0              # regularization parameter
    batch_size = 50         # batch size
    epochs = 100            # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    device = gpu            # default device
    latent_dim = 2          # latent dimension
    hidden_dim = 32         # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    nll = Flux.Losses.mse               # negative log likelihood -log(p(x|z)): MSE for Gaussian, logit binary cross-entropy for Bernoulli
    opt = Adam(η)           # optimizer
end