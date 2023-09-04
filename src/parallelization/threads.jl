"The `ThreadsParallelizer` type is used to parallelize the evaluation of a function using `Threads.@threads`."
struct ThreadsParallelizer <: CounterfactualExplanations.AbstractParallelizer end

function CounterfactualExplanations.parallelize(
    parallelizer::ThreadsParallelizer,
    f::typeof(CounterfactualExplanations.generate_counterfactual),
    args...;
    kwargs...,
)
    @assert CounterfactualExplanations.parallelizable(f) "`f` is not a parallelizable process."

    # Extract positional arguments:
    counterfactuals = args[1] |> x -> CounterfactualExplanations.vectorize_collection(x)
    target = args[2]
    data = args[3]
    M = args[4]
    generator = args[5]

    @assert typeof(M) <: CounterfactualExplanations.AbstractFittedModel ||
        length(M) == length(counterfactuals) "The number of models must match the number of counterfactuals or be a single model."
    @assert typeof(generator) <: CounterfactualExplanations.AbstractGenerator ||
        length(generator) == length(counterfactuals) "The number of generators must match the number of counterfactuals or be a single generator."

    # Zip arguments:
    args = zip(
        counterfactuals,
        fill(target, length(counterfactuals)),
        fill(data, length(counterfactuals)),
        M |> x -> if typeof(M) <: CounterfactualExplanations.AbstractFittedModel
            fill(M, length(counterfactuals))
        else
            M
        end,
        generator |>
        x -> if typeof(generator) <: CounterfactualExplanations.AbstractGenerator
            fill(generator, length(counterfactuals))
        else
            generator
        end,
    )

    ces = []

    Threads.@threads for (x, target, data, M, generator) in collect(args)
        push!(ces, f(x, target, data, M, generator; kwargs...))
    end

    return ces
end
