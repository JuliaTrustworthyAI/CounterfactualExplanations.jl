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
    buffer::Union{Nothing,AbstractArray}
    yidx::Union{Nothing,Any}
end

"""
    EnergySampler(
        model::AbstractModel,
        data::CounterfactualData,
        y::Any;
        opt::AbstractSamplingRule=ImproperSGLD(),
        niter::Int=100,
        nsamples::Int=1000
    )

Constructor for `EnergySampler` that takes a `model`, `data` and conditioning value `y` as inputs.
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
    energy_sampler.buffer = Xpost

    return energy_sampler
end

"""
    EnergySampler(
        ce::CounterfactualExplanation;
        kwrgs...
    )

Constructor for `EnergySampler` that takes a `CounterfactualExplanation` as input. The underlying model, data and `target` are used for the `EnergySampler`, where `target` is the conditioning value of `y`.
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
"""
function train!(e::EnergySampler, y::Int; niter::Int=20, ntransitions::Int=100, kwargs...)

    # Generate samples through persistent contrastive divergence (PCD):
    rule = e.opt

    # Run PCD:
    PCD(e.sampler, e.model, rule; niter=niter, ntransitions=ntransitions, y=y, kwargs...)

    return e
end

"""
    generate_posterior_samples(
        e::EnergySampler, n::Int=1000; niter::Int=500, kwargs...
    )

Uses the replay buffer to generate `n` samples from the posterior distribution.
"""
function generate_posterior_samples(
    e::EnergySampler, n::Int=1000; niter::Int=500, kwargs...
)
    X = e.sampler(e.model, e.opt; n_samples=n, niter=niter, kwargs...)
    return X
end

"""
    Base.rand(sampler::EnergySampler, n::Int=100; retrain=false)

Overloads the `rand` method to randomly draw `n` samples from `EnergySampler`.
"""
function Base.rand(sampler::EnergySampler, n::Int=100; from_buffer=true, niter::Int=500)
    ntotal = size(sampler.buffer, 2)
    idx = rand(1:ntotal, n)
    if from_buffer
        X = sampler.buffer[:, idx]
    else
        X = generate_posterior_samples(sampler, n; niter=niter)
    end
    return X
end

"""
    get_lowest_energy_sample(sampler::EnergySampler; n::Int=5)

Chooses the samples with the lowest energy (i.e. highest probability) from `EnergySampler`.
"""
function get_lowest_energy_sample(sampler::EnergySampler; n::Int=5)
    X = sampler.buffer
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

Define the prior sampling space for the data.
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

Computes the distance from the counterfactual to generated conditional samples.
"""
function distance_from_posterior(
    ce::AbstractCounterfactualExplanation;
    niter::Int=50,
    batch_size::Int=50,
    ntransitions::Int=100,
    prob_buffer::AbstractFloat=0.95,
    nsamples::Int=50,
    niter_final::Int=500,
    from_buffer=true,
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
                kwargs...,
            )
        end
        eng_sampler = _dict[:energy_sampler]
        if choose_lowest_energy
            nmin = minimum([nmin, size(eng_sampler.buffer)[end]])
            xmin = get_lowest_energy_sample(eng_sampler; n=nmin)
            push!(conditional_samples, xmin)
        elseif choose_random
            push!(conditional_samples, rand(eng_sampler, nsamples; from_buffer=from_buffer))
        else
            push!(conditional_samples, eng_sampler.buffer)
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
