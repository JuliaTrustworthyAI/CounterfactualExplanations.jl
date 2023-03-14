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