using Logging

function estimate_memory_usage(
    data::CounterfactualData;
    models::Dict{<:Any,<:Any}=standard_models_catalogue,
    generators::Union{Nothing,Dict{<:Any,<:AbstractGenerator}}=nothing,
    measure::Union{Function,Vector{Function}}=default_measures,
    n_individuals::Int=5,
    suppress_training::Bool=true,
    factual::Union{Nothing,RawTargetType}=nothing,
    target::Union{Nothing,RawTargetType}=nothing,
    nevals::Int=1,
)   

    mem_total = 0

    for i in 1:nevals
        # Setup
        _factual = isnothing(factual) ? rand(data.y_levels) : factual
        _target = isnothing(target) ? rand(data.y_levels[data.y_levels .!= _factual]) : target

        if !suppress_training
            @warn "Estimating memory usage of on-the-fly training currently not supported. Will only report estimated memory usage of inference."
        end
        generators = if isnothing(generators)
            Dict(key => gen() for (key, gen) in generator_catalogue)
        else
            generators
        end

        # Grid setup:
        grid = []
        for (mod_name, M) in models
            # Individuals need to be chosen separately for each model:
            chosen = rand(
                findall(CounterfactualExplanations.predict_label(M, data) .== _factual),
                n_individuals,
            )
            xs = CounterfactualExplanations.select_factual(data, chosen)
            xs = CounterfactualExplanations.vectorize_collection(xs)
            # Form the grid:
            for x in xs
                for (gen_name, gen) in generators
                    comb = (x, (mod_name, M), (gen_name, gen))
                    push!(grid, comb)
                end
            end
        end

        # Vectorize the grid:
        xs = [x[1] for x in grid]
        Ms = [x[2][2] for x in grid]
        gens = [x[3][2] for x in grid]

        if i == 1
            # Warmup:
            ce = with_logger(NullLogger()) do
                generate_counterfactual.([xs[1]], _target, data, Ms, gens)
            end
            benchmark(ce; measure=measure)
        end
        
        # Generating counterfactual:
        mem_each = @allocated begin
            global ce = with_logger(NullLogger()) do
                generate_counterfactual.([xs[1]], _target, data, Ms, gens)
            end
        end

        # Benchmarking counterfactual:
        mem_each += @allocated evaluate.(ce; measure=measure)
        # Total memory usage:
        mem_total += mem_each * n_individuals

    end
    
    return mem_total / nevals

end