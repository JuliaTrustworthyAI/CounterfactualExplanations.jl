"""
    objective(generator, ex)

A macro that can be used to define the counterfactual search objective.
"""
macro objective(generator, ex)
    gen = esc(generator)
    loss = getfield(CounterfactualExplanations.Objectives, ex.args[2])
    Λ = Vector{AbstractFloat}()
    costs = Vector{Function}()
    for i in 3:length(ex.args)
        ex_penalty = ex.args[i]
        λ = ex_penalty.args[2]
        push!(Λ, λ)
        cost = getfield(CounterfactualExplanations.Objectives, ex_penalty.args[3])
        push!(costs, cost)
    end
    ex_generator = quote
        $gen.loss = $loss
        $gen.complexity = $costs
        $gen.λ = $Λ
        $gen
    end
    return ex_generator
end

"""
    threshold(generator, γ)

A simple macro that can be used to define the decision threshold `γ`.
"""
macro threshold(generator, γ)
    return esc(:($generator.decision_threshold = $γ; $generator))
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