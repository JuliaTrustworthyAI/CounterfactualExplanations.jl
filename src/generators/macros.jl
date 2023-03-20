function get_fun(ex)
    if ex ∈ names(CounterfactualExplanations.Objectives)
        return getfield(CounterfactualExplanations.Objectives, ex)
    else
        return getfield(Main, ex)
    end
end

"""
    objective(generator, ex)

A macro that can be used to define the counterfactual search objective.
"""
macro objective(generator, ex)
    gen = esc(generator)
    if ex.args[2] == :_
        loss = nothing
    else
        loss = get_fun(ex.args[2])
    end
    Λ = Vector{AbstractFloat}()
    costs = Vector{Function}()
    for i in 3:length(ex.args)
        ex_penalty = ex.args[i]
        λ = ex_penalty.args[2]
        push!(Λ, λ)
        cost = get_fun(ex_penalty.args[3])
        push!(costs, cost)
    end
    ex_generator = quote
        $gen.loss = $loss
        $gen.penalty = $costs
        $gen.λ = $Λ
        $gen
    end
    return ex_generator
end

"""
    search_latent_space(generator)

A simple macro that can be used to specify latent space search.
"""
macro search_latent_space(generator)
    return esc(:($generator.latent_space = true; $generator))
end

"""
    search_feature_space(generator)

A simple macro that can be used to specify feature space search.
"""
macro search_feature_space(generator)
    return esc(:($generator.feature_space = false; $generator))
end

"""
    with_optimiser(generator, optimiser)

A simple macro that can be used to specify the optimiser to be used.
"""
macro with_optimiser(generator, optimiser)
    return esc(:($generator.opt = $optimiser; $generator))
end
