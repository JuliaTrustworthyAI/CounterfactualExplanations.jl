"""
    h(generator::AbstractGenerator, ce::AbstractCounterfactualExplanation)

Dispatches to the appropriate complexity function for any generator.
"""
function h(generator::AbstractGenerator, ce::AbstractCounterfactualExplanation)
    return h(generator, generator.penalty, ce)
end

"""
    h(generator::AbstractGenerator, penalty::Nothing, ce::AbstractCounterfactualExplanation)

Overloads the `h` function for the case where no penalty is provided.
"""
function h(
    generator::AbstractGenerator, penalty::Nothing, ce::AbstractCounterfactualExplanation
)
    return 0.0
end

"""
    h(generator::AbstractGenerator, penalty::Function, ce::AbstractCounterfactualExplanation)

Overloads the `h` function for the case where a single penalty function is provided.
"""
function h(
    generator::AbstractGenerator, penalty::Function, ce::AbstractCounterfactualExplanation
)
    return generator.λ .* penalty(ce)
end

"""
    h(generator::AbstractGenerator, penalty::Tuple, ce::AbstractCounterfactualExplanation)

Overloads the `h` function for the case where a single penalty function is provided with additional keyword arguments.
"""
function h(
    generator::AbstractGenerator,
    penalty::Vector{Function},
    ce::AbstractCounterfactualExplanation,
)
    return sum(generator.λ .* [fun(ce) for fun in penalty])
end

"""
    h(generator::AbstractGenerator, penalty::Tuple, ce::AbstractCounterfactualExplanation)

Overloads the `h` function for the case where a single penalty function is provided with additional keyword arguments.
"""
function h(
    generator::AbstractGenerator,
    penalty::Vector{<:Tuple},
    ce::AbstractCounterfactualExplanation,
)
    return sum(generator.λ .* [fun(ce; kwargs...) for (fun, kwargs) in penalty])
end
