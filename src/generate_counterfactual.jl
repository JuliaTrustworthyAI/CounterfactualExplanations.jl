# -------- Main method:
"""
    generate_counterfactuals(
        x::AbstractArray,
        n::Int=1;
        target::Union{AbstractFloat,Int},
        data::CounterfactualData,
        M::Models.AbstractFittedModel,
        generator::AbstractGenerator,
        T::Int = 1000,
        latent_space::Union{Nothing,Bool} = nothing,
        num_counterfactuals::Int = 1,
        initialization::Symbol = :add_perturbation,
        generative_model_params::NamedTuple = (;),
    )

The core function that is used to run counterfactual search for a given factual `x`, target, counterfactual data, model and generator. Keywords can be used to specify the desired threshold for the predicted target class probability and the maximum number of iterations.
"""
function generate_counterfactuals(
    x::AbstractArray,
    n::Int=1;
    target::Union{AbstractFloat,Int},
    data::CounterfactualData,
    M::Models.AbstractFittedModel,
    generator::AbstractGenerator,
    T::Int=1000,
    latent_space::Union{Nothing,Bool}=nothing,
    num_counterfactuals::Int=1,
    initialization::Symbol=:add_perturbation,
    generative_model_params::NamedTuple=(;)
)

    results = []

    for i in 1:n

        # Initialize:
        ce = CounterfactualExplanation(
            x=x,
            target=target,
            data=data,
            M=M,
            generator=generator,
            T=T,
            latent_space=latent_space,
            num_counterfactuals=num_counterfactuals,
            initialization=initialization,
            generative_model_params=generative_model_params,
        )

        # Search:
        while !ce.search[:terminated]
            update!(ce)
        end

        push!(results, ce)

    end

    return results

end


function generate_counterfactuals(
    x::Base.Iterators.Zip,
    n::Int;
    kwargs...
)
    results =
        map(x_ -> generate_counterfactuals(x_[1], n; kwargs...), x)

    return results

end
