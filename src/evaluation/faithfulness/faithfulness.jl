include("utils.jl")

"""
    distance_from_energy(ce::AbstractCounterfactualExplanation)

Computes the distance from the counterfactual to generated conditional samples.
"""
function distance_from_energy(
    ce::AbstractCounterfactualExplanation;
    n::Int=50,
    niter=500,
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
    nmin = minimum([nmin, n])

    @assert choose_lowest_energy ⊻ choose_random || !choose_lowest_energy && !choose_random "Must choose either lowest energy or random samples or neither."

    conditional_samples = []
    ignore_derivatives() do
        _dict = ce.params
        if !(:energy_sampler ∈ collect(keys(_dict)))
            _dict[:energy_sampler] = EnergySampler(ce; niter=niter, nsamples=n, kwargs...)
        end
        eng_sampler = _dict[:energy_sampler]
        if choose_lowest_energy
            nmin = minimum([nmin, size(eng_sampler.buffer)[end]])
            xmin = get_lowest_energy_sample(eng_sampler; n=nmin)
            push!(conditional_samples, xmin)
        elseif choose_random
            push!(conditional_samples, rand(eng_sampler, n; from_buffer=from_buffer))
        else
            push!(conditional_samples, eng_sampler.buffer)
        end
    end

    _loss = map(eachcol(conditional_samples[1])) do xsample
        distance(ce; from=xsample, agg=agg, p=p)
    end
    _loss = reduce((x, y) -> x + y, _loss) / n       # aggregate over samples

    if return_conditionals
        return conditional_samples[1]
    end
    return _loss
end
