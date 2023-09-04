function make_grid(
    xs::Union{AbstractArray,Base.Iterators.Zip},
    target::RawTargetType,
    data::CounterfactualData,
    models::Dict{<:Any,<:AbstractFittedModel},
    generators::Dict{<:Any,<:AbstractGenerator},
)
    grid = Base.Iterators.product([xs], [target], [data], models, generators)
    return grid
end

function make_grid(
    factual::RawTargetType,
    target::RawTargetType,
    data::CounterfactualData,
    models::Dict{<:Any,<:AbstractFittedModel},
    generators::Dict{<:Any,<:AbstractGenerator},
    n_individuals::Int=5,
)

    grid = []
    for (mod_name, M) in models
        # Individuals need to be chosen separately for each model:
        chosen = rand(
            findall(CounterfactualExplanations.predict_label(M, data) .== factual),
            n_individuals,
        )
        xs = CounterfactualExplanations.select_factual(data, chosen)
        xs = CounterfactualExplanations.vectorize_collection(xs)
        # Form the grid:
        for x in xs
            for (gen_name, gen) in generators
                comb = (x, target, data, (mod_name, M), (gen_name, gen))
                push!(grid, comb)
            end
        end
    end
    
    return Iterators.flatten(zip(grid))
end