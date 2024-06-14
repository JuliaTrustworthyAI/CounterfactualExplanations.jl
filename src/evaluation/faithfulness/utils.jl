using CounterfactualExplanations
using CounterfactualExplanations.Models
using ChainRulesCore: ChainRulesCore
using Distributions
using Flux
using TaijaBase: Samplers
using TaijaBase.Samplers: SGLD, ImproperSGLD, ConditionalSampler, AbstractSamplingRule, PMC

"Base type that stores information relevant to energy-based posterior sampling from `AbstractModel`."
mutable struct EnergySampler
    model::AbstractModel
    sampler::ConditionalSampler
    opt::AbstractSamplingRule
    posterior::Union{Nothing,AbstractArray}
    yidx::Union{Nothing,Any}
end

"""
    EnergySampler(
        model::AbstractModel,
        𝒟x::Distribution,
        𝒟y::Distribution,
        input_size::Dims,
        yidx::Int;
        opt::Union{Nothing,AbstractSamplingRule}=nothing,
        nsamples::Int=100,
        niter_final::Int=1000,
        ntransitions::Int=0,
        opt_warmup::Union{Nothing,AbstractSamplingRule}=nothing,
        niter::Int=20,
        batch_size::Int=50,
        prob_buffer::AbstractFloat=0.95,
        kwargs...,
    )

Constructor for `EnergySampler`, which is used to sample from the posterior distribution of the model conditioned on `y`.

# Arguments

- `model::AbstractModel`: The model to be used for sampling.
- `data::CounterfactualData`: The data to be used for sampling.
- `y::Any`: The conditioning value.
- `opt::AbstractSamplingRule=ImproperSGLD()`: The sampling rule to be used. By default, `SGLD` is used with `a = (2 / std(Uniform()) * std(𝒟x)` and `b = 1` and `γ=0.9`.
- `nsamples::Int=100`: The number of samples to include in the final empirical posterior distribution.
- `niter_final::Int=1000`: The number of iterations for generating samples from the posterior distribution. Typically, this number will be larger than the number of iterations during PMC training. 
- `ntransitions::Int=0`: The number of transitions for (optionally) warming up the sampler. By default, this is set to 0 and the sampler is not warmed up. For valies larger than 0, the sampler is trained through PMC for `niter` iterations and `ntransitions` transitions to build a buffer of samples. The buffer is used for posterior sampling.
- `opt_warmup::Union{Nothing,AbstractSamplingRule}=nothing`: The sampling rule to be used for warm-up. By default, `ImproperSGLD` is used with `α = (2 / std(Uniform()) * std(𝒟x)` and `γ = 0.005α`.
- `niter::Int=100`: The number of iterations for training the sampler through PMC.
- `batch_size::Int=50`: The batch size for training the sampler.
- `prob_buffer::AbstractFloat=0.5`: The probability of drawing samples from the replay buffer. Smaller values will result in more samples being drawn from the prior and typically lead to better mixing and diversity in the samples.
- `kwargs...`: Additional keyword arguments to be passed on to the sampler and PMC.

# Returns

- `EnergySampler`: An instance of `EnergySampler`.
"""
function EnergySampler(
    model::AbstractModel,
    𝒟x::Distribution,
    𝒟y::Distribution,
    input_size::Dims,
    yidx::Int;
    opt::Union{Nothing,AbstractSamplingRule}=nothing,
    nsamples::Int=100,
    niter_final::Int=1000,
    batch_size_final::Int=round(Int, nsamples / 100),
    nwarmup::Int=0,
    opt_warmup::Union{Nothing,AbstractSamplingRule}=nothing,
    niter::Int=20,
    batch_size::Int=50,
    prob_buffer::AbstractFloat=0.0,
    kwargs...,
)

    # Sampler:
    sampler = ConditionalSampler(
        𝒟x, 𝒟y; input_size=input_size, prob_buffer=prob_buffer, batch_size=batch_size
    )

    # Optimizer:
    if isnothing(opt)
        α = (2 / std(Uniform())) * std(𝒟x)
        b = round(Int, niter_final / 100)

        opt = SGLD(; a=α, b=b, γ=0.9)
    end

    # Initiate:
    energy_sampler = EnergySampler(model, sampler, opt, [], yidx)

    # Warm-up sampler:
    if nwarmup > 0
        @info "Warming up sampler..."
        ntransitions = round(Int, nwarmup / batch_size)
        println(ntransitions)
        warmup!(
            energy_sampler,
            yidx;
            opt=opt_warmup,
            niter=niter,
            ntransitions=ntransitions,
            kwargs...,
        )
    end

    # Construct posterior samples:
    @info "Generating posterior samples..."
    Xpost = generate_posterior_samples(
        energy_sampler, nsamples; batch_size=batch_size_final, niter=niter_final, kwargs...
    )
    energy_sampler.posterior = Xpost

    return energy_sampler
end

"""
    define_prior(
        data::CounterfactualData;
        𝒟x::Union{Nothing,Distribution}=nothing,
        𝒟y::Union{Nothing,Distribution}=nothing,
        n_std::Int=3,
    )

Defines the prior for the data. The space is defined as a uniform distribution with bounds defined by the mean and standard deviation of the data. The bounds are extended by `n_std` standard deviations.

# Arguments

- `data::CounterfactualData`: The data to be used for defining the prior sampling space.
- `n_std::Int=3`: The number of standard deviations to extend the bounds.

# Returns

- `Uniform`: The uniform distribution defining the prior sampling space.
"""
function define_prior(
    data::CounterfactualData;
    𝒟x::Union{Nothing,Distribution}=nothing,
    𝒟y::Union{Nothing,Distribution}=nothing,
    n_std::Int=2,
)

    # Input space:
    X = data.X
    centers = mean(X; dims=2)
    stds = std(X; dims=2)
    lower_bound = minimum(centers .- n_std .* stds)[1]
    upper_bound = maximum(centers .+ n_std .* stds)[1]
    𝒟x = isnothing(𝒟x) ? Uniform(lower_bound, upper_bound) : 𝒟x

    # Output space:
    K = length(data.y_levels)
    𝒟y = isnothing(𝒟y) ? Categorical(ones(K) ./ K) : 𝒟y

    return 𝒟x, 𝒟y
end

"""
    EnergySampler(ce::CounterfactualExplanation; kwrgs...)

Overloads the `EnergySampler` constructor to accept a `CounterfactualExplanation` object.
"""
function EnergySampler(ce::CounterfactualExplanation; kwrgs...)

    # Setup:
    model = ce.M

    # Target index:
    y = ce.target
    yidx = get_target_index(ce.data.y_levels, y)

    # Input size:
    input_size = size(selectdim(ce.data.X, ndims(ce.data.X), 1))

    # Define prior:
    𝒟x, 𝒟y, = define_prior(ce.data)

    # Construct:
    return EnergySampler(model, 𝒟x, 𝒟y, input_size, yidx; kwrgs...)
end

"""
    warmup!(
        e::EnergySampler,
        y::Int;
        niter::Int=20,
        ntransitions::Int=100,
        kwargs...,
    )

Warms up the `EnergySampler` to the underlying model for conditioning value `y`. Specifically, this entails running PMC for `niter` iterations and `ntransitions` transitions to build a buffer of samples. The buffer is used for posterior sampling.

# Arguments

- `e::EnergySampler`: The `EnergySampler` object to be trained.
- `y::Int`: The conditioning value.
- `opt::Union{Nothing,AbstractSamplingRule}`: The sampling rule to be used. By default, `ImproperSGLD` is used with `α = 2 * std(Uniform(𝒟x))` and `γ = 0.005α`.
- `niter::Int=20`: The number of iterations for training the sampler through PMC.
- `ntransitions::Int=100`: The number of transitions for training the sampler. In each transition, the sampler is updated with a mini-batch of data. Data is either drawn from the replay buffer or reinitialized from the prior.
- `kwargs...`: Additional keyword arguments to be passed on to the sampler and PMC.

# Returns

- `EnergySampler`: The trained `EnergySampler`.
"""
function warmup!(
    e::EnergySampler,
    y::Int;
    opt::Union{Nothing,AbstractSamplingRule},
    niter::Int=20,
    ntransitions::Int=100,
    kwargs...,
)

    # Set up sampling rule:
    if isnothing(opt)
        α = (2 / std(Uniform())) * std(e.sampler.𝒟x)
        σ = 0.005 * α
        rule = ImproperSGLD(α, σ)
    else
        rule = opt
    end

    # Run PMC with improper SGLD:
    PMC(
        e.sampler,
        e.model,
        rule;
        niter=niter,
        ntransitions=ntransitions,
        y=y,
        clip_grads=nothing,
        kwargs...,
    )

    return e
end

"""
    generate_posterior_samples(
        e::EnergySampler, n::Int=1000; niter::Int=1000, kwargs...
    )

Generates `n` samples from the posterior distribution of the model conditioned on the target value `y`. The samples are generated through (Persistent) Monte Carlo sampling using the `EnergySampler` object. If the replay buffer is not empty, the initial samples are drawn from the buffer. 

Note that by default the batch size of the sampler is set to `round(Int, n / 100)` by default for sampling. This is to ensure that the samples are drawn independently from the posterior distribution. It also helps to avoid vanishing gradients. 

The chain is run persistently until `n` samples are generated. The number of transitions is set to `ceil(Int, n / batch_size)`. Once the chain is run, the last `n` samples are form the replay buffer are returned.

# Arguments

- `e::EnergySampler`: The `EnergySampler` object to be used for sampling.
- `n::Int=100`: The number of samples to generate.
- `batch_size::Int=round(Int, n / 100)`: The batch size for sampling.
- `niter::Int=1000`: The number of iterations for generating samples from the posterior distribution.
- `kwargs...`: Additional keyword arguments to be passed on to the sampler.

# Returns

- `AbstractArray`: The generated samples.
"""
function generate_posterior_samples(
    e::EnergySampler, n::Int=1000; batch_size=round(Int, n / 10), niter::Int=1000, kwargs...
)

    # Store batch size:
    bs = e.sampler.batch_size

    # Specify batch size for sampling:
    batch_size = maximum([1, batch_size])            # ensure batch size is at least 1
    e.sampler.batch_size = batch_size               # set batch size for sampling
    ntransitions = ceil(Int, n / batch_size) + 1    # number of transitions to generate n samples

    # Adjust buffer size if necessary:
    e.sampler.max_len = maximum([e.sampler.max_len, n])

    # Generate samples through (Persistent) Monte Carlo sampling:
    X = PMC(
        e.sampler,
        e.model,
        e.opt;
        ntransitions=ntransitions,
        niter=niter,
        y=e.yidx,
        clip_grads=nothing,
        kwargs...,
    )[
        :, 1:n
    ]

    # Reset batch size:
    e.sampler.batch_size = bs

    return X
end

"""
    get_sampler!(ce::AbstractCounterfactualExplanation; kwargs...)

Gets the `EnergySampler` object from the counterfactual explanation. If the sampler is not found, it is constructed and stored in the counterfactual explanation object.
"""
function get_sampler!(ce::AbstractCounterfactualExplanation; kwargs...)

    # Get full dictionary:
    get!(ce.search, :energy_sampler) do
        get!(ce.M.fitresult.other, :energy_sampler) do
            Dict()
        end
    end

    # Get sampler at target index:
    y = ce.target
    get!(ce.search[:energy_sampler], y) do
        get!(ce.M.fitresult.other[:energy_sampler], y) do
            EnergySampler(ce; kwargs...)
        end
    end
end

"""
    distance_from_posterior(ce::AbstractCounterfactualExplanation)

Computes the distance from the counterfactual to generated conditional samples. The distance is computed as the mean distance from the counterfactual to the samples drawn from the posterior distribution of the model. 

# Arguments

- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation object.
- `nsamples::Int=1000`: The number of samples to draw.
- `from_posterior::Bool=true`: Whether to draw samples from the posterior distribution.
- `agg`: The aggregation function to use for computing the distance.
- `choose_lowest_energy::Bool=true`: Whether to choose the samples with the lowest energy.
- `choose_random::Bool=false`: Whether to choose random samples.
- `nmin::Int=25`: The minimum number of samples to choose.
- `p::Int=1`: The norm to use for computing the distance.
- `kwargs...`: Additional keyword arguments to be passed on to the [`EnergySampler`](@ref).

# Returns

- `AbstractFloat`: The distance from the counterfactual to the samples.
"""
function distance_from_posterior(
    ce::AbstractCounterfactualExplanation;
    nsamples::Int=1000,
    from_posterior=true,
    agg=mean,
    choose_lowest_energy=false,
    choose_random=false,
    nmin::Int=25,
    p::Int=1,
    kwargs...,
)
    _loss = 0.0
    nmin = minimum([nmin, nsamples])

    @assert choose_lowest_energy ⊻ choose_random || !choose_lowest_energy && !choose_random "Must choose either lowest energy or random samples or neither."

    # Get energy sampler from model:
    smpler = get_sampler!(ce; nsamples=nsamples, kwargs...)

    # Get conditional samples from posterior:
    conditional_samples = []
    ChainRulesCore.ignore_derivatives() do
        if choose_lowest_energy
            nmin = minimum([nmin, size(smpler.posterior)[end]])
            xmin = get_lowest_energy_sample(smpler; n=nmin)
            push!(conditional_samples, xmin)
        elseif choose_random
            push!(
                conditional_samples, rand(smpler, nsamples; from_posterior=from_posterior)
            )
        else
            push!(conditional_samples, smpler.posterior)
        end
    end

    # Compute distance:
    _loss = map(eachcol(conditional_samples[1])) do xsample
        distance(ce; from=xsample, agg=agg, p=p)
    end
    _loss = reduce((x, y) -> x + y, _loss) / nsamples       # aggregate over samples

    return _loss
end

"""
    Base.rand(sampler::EnergySampler, n::Int=100; retrain=false)

Overloads the `rand` method to randomly draw `n` samples from `EnergySampler`. If `from_posterior` is `true`, the samples are drawn from the posterior distribution. Otherwise, the samples are generated from the model conditioned on the target value using a single chain (see [`generate_posterior_samples`](@ref)).

# Arguments

- `sampler::EnergySampler`: The `EnergySampler` object to be used for sampling.
- `n::Int=100`: The number of samples to draw.
- `from_posterior::Bool=true`: Whether to draw samples from the posterior distribution.
- `niter::Int=500`: The number of iterations for generating samples through Monte Carlo sampling (single chain).

# Returns

- `AbstractArray`: The samples.
"""
function Base.rand(sampler::EnergySampler, n::Int=100; from_posterior=true, niter::Int=500)
    ntotal = size(sampler.posterior, 2)
    idx = rand(1:ntotal, n)
    if from_posterior
        X = sampler.posterior[:, idx]
    else
        X = generate_posterior_samples(sampler, n; niter=niter)
    end
    return X
end

"""
    get_lowest_energy_sample(sampler::EnergySampler; n::Int=5)

Chooses the samples with the lowest energy (i.e. highest probability) from `EnergySampler`.

# Arguments

- `sampler::EnergySampler`: The `EnergySampler` object to be used for sampling.
- `n::Int=5`: The number of samples to choose.

# Returns

- `AbstractArray`: The samples with the lowest energy.
"""
function get_lowest_energy_sample(sampler::EnergySampler; n::Int=5)
    X = sampler.posterior
    model = sampler.model
    y = sampler.yidx
    x = selectdim(
        X,
        ndims(X),
        Samplers.energy(sampler.sampler, model, X, y; agg=x -> partialsortperm(x, 1:n)),
    )
    return x
end
