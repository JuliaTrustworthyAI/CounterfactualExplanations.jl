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
    data::CounterfactualData
    sampler::ConditionalSampler
    opt::AbstractSamplingRule
    posterior::Union{Nothing,AbstractArray}
    yidx::Union{Nothing,Any}
end

"""
    EnergySampler(
        model::AbstractModel,
        data::CounterfactualData,
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
    data::CounterfactualData,
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
    @assert y ∈ data.y_levels || y ∈ 1:length(data.y_levels)

    K = length(data.y_levels)
    input_size = size(selectdim(data.X, ndims(data.X), 1))

    # Prior distribution:
    𝒟x = prior_sampling_space(data)
    𝒟y = Categorical(ones(K) ./ K)

    # Sampler:
    sampler = ConditionalSampler(
        𝒟x, 𝒟y; input_size=input_size, prob_buffer=prob_buffer, batch_size=batch_size
    )
    yidx = get_target_index(data.y_levels, y)

    # Initiate:
    energy_sampler = EnergySampler(model, data, sampler, opt, nothing, yidx)

    # Train:
    train!(energy_sampler, yidx; niter=niter, ntransitions=ntransitions, kwargs...)

    # Construct posterior samples:
    Xpost = generate_posterior_samples(
        energy_sampler, nsamples; niter=niter_final, kwargs...
    )
    energy_sampler.posterior = Xpost

    return energy_sampler
end

"""
    EnergySampler(ce::CounterfactualExplanation; kwrgs...)

Overloads the `EnergySampler` constructor to accept a `CounterfactualExplanation` object.
"""
function EnergySampler(ce::CounterfactualExplanation; kwrgs...)

    # Setup:
    model = ce.M
    data = ce.data
    y = ce.target

    return EnergySampler(model, data, y; kwrgs...)
end

"""
    train!(
        e::EnergySampler,
        y::Int;
        niter::Int=20,
        ntransitions::Int=100,
        kwargs...,
    )

Trains the `EnergySampler` for conditioning value `y`. Specifically, this entails running PCD for `niter` iterations and `ntransitions` transitions to build a buffer of samples. The buffer is used for posterior sampling.

# Arguments

- `e::EnergySampler`: The `EnergySampler` object to be trained.
- `y::Int`: The conditioning value.
- `niter::Int=20`: The number of iterations for training the sampler through PCD.
- `ntransitions::Int=100`: The number of transitions for training the sampler. In each transition, the sampler is updated with a mini-batch of data. Data is either drawn from the replay buffer or reinitialized from the prior.
- `kwargs...`: Additional keyword arguments to be passed on to the sampler and PCD.

# Returns

- `EnergySampler`: The trained `EnergySampler`.
"""
function train!(e::EnergySampler, y::Int; niter::Int=20, ntransitions::Int=100, kwargs...)

    # Generate samples through persistent contrastive divergence (PCD):
    rule = e.opt

    # Run PCD:
    PCD(e.sampler, e.model, ImproperSGLD(); niter=niter, ntransitions=ntransitions, y=nothing, kwargs...)

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
    prior_sampling_space(data::CounterfactualData; n_std=3)

Defines the prior sampling space for the data. The space is defined as a uniform distribution with bounds defined by the mean and standard deviation of the data. The bounds are extended by `n_std` standard deviations.

# Arguments

- `data::CounterfactualData`: The data to be used for defining the prior sampling space.
- `n_std::Int=3`: The number of standard deviations to extend the bounds.

# Returns

- `Uniform`: The uniform distribution defining the prior sampling space.
"""
function prior_sampling_space(data::CounterfactualData; n_std=3)
    X = data.X
    centers = mean(X; dims=2)
    stds = std(X; dims=2)
    lower_bound = minimum(centers .- n_std .* stds)[1]
    upper_bound = maximum(centers .+ n_std .* stds)[1]
    return Uniform(lower_bound, upper_bound)
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
            push!(conditional_samples, rand(eng_sampler, nsamples; from_posterior=from_posterior))
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
