using CounterfactualExplanations
using CounterfactualExplanations.Models
using ChainRulesCore: ChainRulesCore
using Distributions
using Flux
using TaijaBase: Samplers
using TaijaBase.Samplers: SGLD, ImproperSGLD, ConditionalSampler, AbstractSamplingRule, PCD

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
        y::Any;
        opt::AbstractSamplingRule=ImproperSGLD(2.0, 0.01),
        niter::Int=20,
        batch_size::Int=50,
        ntransitions::Int=100,
        prob_buffer::AbstractFloat=0.95,
        nsamples::Int=50,
        niter_final::Int=500,
        kwargs...,
    )

Constructor for `EnergySampler`, which is used to sample from the posterior distribution of the model conditioned on `y`.

# Arguments

- `model::AbstractModel`: The model to be used for sampling.
- `data::CounterfactualData`: The data to be used for sampling.
- `y::Any`: The conditioning value.
- `opt::AbstractSamplingRule=ImproperSGLD()`: The sampling rule to be used.
- `niter::Int=100`: The number of iterations for training the sampler through PCD.
- `batch_size::Int=50`: The batch size for training the sampler.
- `ntransitions::Int=100`: The number of transitions for training the sampler. In each transition, the sampler is updated with a mini-batch of data. Data is either drawn from the replay buffer or reinitialized from the prior.
- `prob_buffer::AbstractFloat=0.95`: The probability of drawing samples from the replay buffer.
- `nsamples::Int=50`: The number of samples to include in the final empirical posterior distribution.
- `niter_final::Int=500`: The number of iterations for generating samples from the posterior distribution.
- `kwargs...`: Additional keyword arguments to be passed on to the sampler and PCD.

# Returns

- `EnergySampler`: An instance of `EnergySampler`.
"""
function EnergySampler(
    model::AbstractModel,
    𝒟x::Distribution,
    𝒟y::Distribution,
    input_size::Dims,
    yidx::Int;
    opt::AbstractSamplingRule=ImproperSGLD(2.0, 0.01),
    niter::Int=20,
    batch_size::Int=50,
    ntransitions::Int=100,
    prob_buffer::AbstractFloat=0.95,
    nsamples::Int=50,
    niter_final::Int=500,
    kwargs...,
)

    # Sampler:
    sampler = ConditionalSampler(
        𝒟x, 𝒟y; input_size=input_size, prob_buffer=prob_buffer, batch_size=batch_size
    )

    # Initiate:
    energy_sampler = EnergySampler(model, sampler, opt, nothing, yidx)

    # Train:
    fit!(energy_sampler, yidx; niter=niter, ntransitions=ntransitions, kwargs...)

    # Construct posterior samples:
    Xpost = generate_posterior_samples(
        energy_sampler, nsamples; niter=niter_final, kwargs...
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
    n_std::Int=3,
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
    fit!(
        e::EnergySampler,
        y::Int;
        niter::Int=20,
        ntransitions::Int=100,
        kwargs...,
    )

Fits the `EnergySampler` to the underlying model for conditioning value `y`. Specifically, this entails running PCD for `niter` iterations and `ntransitions` transitions to build a buffer of samples. The buffer is used for posterior sampling.

# Note

For fitting the sampler, `ImproperSGLD` is used as the default sampling rule. The rule is defined as `α = 2 * std(e.sampler.𝒟x)` and `σ = 0.005 * α`.

# Arguments

- `e::EnergySampler`: The `EnergySampler` object to be trained.
- `y::Int`: The conditioning value.
- `niter::Int=20`: The number of iterations for training the sampler through PCD.
- `ntransitions::Int=100`: The number of transitions for training the sampler. In each transition, the sampler is updated with a mini-batch of data. Data is either drawn from the replay buffer or reinitialized from the prior.
- `kwargs...`: Additional keyword arguments to be passed on to the sampler and PCD.

# Returns

- `EnergySampler`: The trained `EnergySampler`.
"""
function fit!(e::EnergySampler, y::Int; niter::Int=20, ntransitions::Int=100, kwargs...)

    # Set up sampling rule:
    α = (2 / std(Uniform())) * std(e.sampler.𝒟x)
    σ = 0.005 * α
    rule = ImproperSGLD(α, σ)
    println(rule)

    # Run PCD with improper SGLD:
    PCD(e.sampler, e.model, rule; niter=niter, ntransitions=ntransitions, y=y, kwargs...)

    # Set probabibility of drawing from buffer to 1 for posterior sampling:
    e.sampler.prob_buffer = 1.0

    return e
end

"""
    generate_posterior_samples(
        e::EnergySampler, n::Int=1000; niter::Int=500, kwargs...
    )

Uses the replay buffer to generate `n` samples from the posterior distribution. Specifically, this entails running a single chain of the sampler for `niter` iterations. Typically, the number of iterations will be larger than during PCD training.

# Arguments

- `e::EnergySampler`: The `EnergySampler` object to be used for sampling.
- `n::Int=1000`: The number of samples to generate.
- `niter::Int=500`: The number of iterations for generating samples from the posterior distribution.
- `kwargs...`: Additional keyword arguments to be passed on to the sampler.

# Returns

- `AbstractArray`: The generated samples.
"""
function generate_posterior_samples(
    e::EnergySampler, n::Int=1000; niter::Int=500, kwargs...
)
    X = e.sampler(e.model, e.opt; n_samples=n, niter=niter, y=e.yidx, kwargs...)
    return X
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

"""
    distance_from_posterior(ce::AbstractCounterfactualExplanation)

Computes the distance from the counterfactual to generated conditional samples. The distance is computed as the mean distance from the counterfactual to the samples drawn from the posterior distribution of the model. 

# Arguments

- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation object.
- `niter::Int=50`: The number of iterations for training the sampler through PCD.
- `batch_size::Int=50`: The batch size for training the sampler.
- `ntransitions::Int=100`: The number of transitions for training the sampler. In each transition, the sampler is updated with a mini-batch of data. Data is either drawn from the replay buffer or reinitialized from the prior.
- `prob_buffer::AbstractFloat=0.95`: The probability of drawing samples from the replay buffer.
- `nsamples::Int=50`: The number of samples to include in the final empirical posterior distribution.
- `niter_final::Int=500`: The number of iterations for generating samples from the posterior distribution.
- `from_posterior::Bool=true`: Whether to draw samples from the posterior distribution.
- `agg`: The aggregation function to use for computing the distance.
- `choose_lowest_energy::Bool=true`: Whether to choose the samples with the lowest energy.
- `choose_random::Bool=false`: Whether to choose random samples.
- `nmin::Int=25`: The minimum number of samples to choose.
- `return_conditionals::Bool=false`: Whether to return the conditional samples.
- `p::Int=1`: The norm to use for computing the distance.
- `kwargs...`: Additional keyword arguments to be passed on to the sampler and PCD.

# Returns

- `AbstractFloat`: The distance from the counterfactual to the samples.
"""
function distance_from_posterior(
    ce::AbstractCounterfactualExplanation;
    niter::Int=50,
    batch_size::Int=50,
    ntransitions::Int=100,
    prob_buffer::AbstractFloat=0.95,
    opt::AbstractSamplingRule=ImproperSGLD(2.0, 0.01),
    nsamples::Int=50,
    niter_final::Int=500,
    from_posterior=true,
    agg=mean,
    choose_lowest_energy=true,
    choose_random=false,
    nmin::Int=25,
    return_conditionals=false,
    p::Int=1,
    kwargs...,
)
    _loss = 0.0
    nmin = minimum([nmin, nsamples])

    @assert choose_lowest_energy ⊻ choose_random || !choose_lowest_energy && !choose_random "Must choose either lowest energy or random samples or neither."

    conditional_samples = []
    ChainRulesCore.ignore_derivatives() do
        _dict = ce.search
        if !(:energy_sampler ∈ collect(keys(_dict)))
            _dict[:energy_sampler] = EnergySampler(
                ce;
                niter=niter,
                batch_size=batch_size,
                ntransitions=ntransitions,
                prob_buffer=prob_buffer,
                nsamples=nsamples,
                niter_final=niter_final,
                opt=opt,
                kwargs...,
            )
        end
        eng_sampler = _dict[:energy_sampler]
        if choose_lowest_energy
            nmin = minimum([nmin, size(eng_sampler.posterior)[end]])
            xmin = get_lowest_energy_sample(eng_sampler; n=nmin)
            push!(conditional_samples, xmin)
        elseif choose_random
            push!(
                conditional_samples,
                rand(eng_sampler, nsamples; from_posterior=from_posterior),
            )
        else
            push!(conditional_samples, eng_sampler.posterior)
        end
    end

    _loss = map(eachcol(conditional_samples[1])) do xsample
        distance(ce; from=xsample, agg=agg, p=p)
    end
    _loss = reduce((x, y) -> x + y, _loss) / nsamples       # aggregate over samples

    if return_conditionals
        return conditional_samples[1]
    end
    return _loss
end
