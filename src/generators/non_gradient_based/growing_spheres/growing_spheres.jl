using LinearAlgebra, Random

"Growing Spheres counterfactual generator class."
mutable struct GrowingSpheresGenerator <: AbstractNonGradientBasedGenerator
    n::Union{Nothing,Integer}
    η::Union{Nothing,AbstractFloat}
end

"""
    GrowingSpheresGenerator(; n::Int=100, η::Float64=0.1, kwargs...)

Constructs a new Growing Spheres Generator object.
"""
# function GrowingSpheresGenerator(; n::Integer=100, η::AbstractFloat=0.1, kwargs...)
#     return GrowingSpheresGenerator(; n=n, η=η, kwargs...)
# end

"""
    hyper_sphere_coordinates(n_search_samples::Int, instance::Vector{Float64}, low::Int, high::Int; p_norm::Int=2)

    Generates candidate counterfactuals using the growing spheres method based on hyper-sphere coordinates.

    The implementation follows the Random Point Picking over a sphere algorithm described in the paper:
    "Learning Counterfactual Explanations for Tabular Data" by Pawelczyk, Broelemann & Kascneci (2020),
    presented at The Web Conference 2020 (WWW). It ensures that points are sampled uniformly at random
    using insights from: http://mathworld.wolfram.com/HyperspherePointPicking.html

    The growing spheres method is originally proposed in the paper:
    "Comparison-based Inverse Classification for Interpretability in Machine Learning" by Thibaut Laugel et al (2018),
    presented at the International Conference on Information Processing and Management of Uncertainty in Knowledge-Based Systems (2018).

    # Arguments
    - n_search_samples::Int64: The number of search samples (int > 0).
    - instance::Array: The Julia input point array.
    - low::Float64: The lower bound (float >= 0, l < h).
    - high::Float64: The upper bound (float >= 0, h > l).
    - p_norm::Float64: The norm parameter (float >= 1).

    # Returns
    - candidate_counterfactuals: An array of candidate counterfactuals.
    - dist: An array of distances corresponding to each candidate counterfactual.
"""
function hyper_sphere_coordinates(n_search_samples::Integer, instance::AbstractArray, low::AbstractFloat, high::AbstractFloat; p_norm::Integer=2)
    delta_instance = randn(n_search_samples, length(instance))
    dist = rand(n_search_samples) .* (high - low) .+ low  # length range [l, h)
    norm_p = LinearAlgebra.norm(delta_instance, p_norm)
    d_norm = dist ./ norm_p  # rescale/normalize factor
    delta_instance .= delta_instance .* d_norm
    instance_matrix = repeat(reshape(instance, 1, length(instance)), n_search_samples)
    candidate_counterfactuals = instance_matrix + delta_instance

    return transpose(candidate_counterfactuals)
end

function growing_spheres_generation(
    generator::GrowingSpheresGenerator,
    model::AbstractFittedModel,
    factual::AbstractArray,
    counterfactual_data::CounterfactualData
)
    # Copy hyperparameters
    n = generator.n
    η = generator.η

    # Generate random points uniformly on a sphere
    counterfactual_candidates = hyper_sphere_coordinates(n, factual, 0.0, η)
    factual_class = CounterfactualExplanations.Models.predict_label(model, counterfactual_data, factual)

    counterfactual = findfirst(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e) ≠ factual_class, eachcol(counterfactual_candidates))
    stop = 100

    # Repeat until there's at least one counterfactual (process of removing them by reducing the search space)
    while(!isnothing(counterfactual) && stop > 0)
        η = η / 2

        counterfactual_candidates = hyper_sphere_coordinates(n, factual, 0.0, η)
        counterfactual = findfirst(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e) ≠ factual_class, eachcol(counterfactual_candidates))
        stop -= 1
    end

    # We expect there're only factual labels:
    @info("Predicted labels: ", map(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e) ≠ factual_class, eachcol(counterfactual_candidates)))
    
    # Initialize boundaries of the spehere's radius
    a₀ = η
    a₁ = 2η

    counterfactual = findfirst(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e) ≠ factual_class, eachcol(counterfactual_candidates))

    # Repeat until there's the first (the closest) counterfactual (process of expanding the search space)
    while(isnothing(counterfactual) && stop > 0)
        a₀ = a₁
        a₁ = a₁ + η

        counterfactual_candidates = hyper_sphere_coordinates(n, factual, 0.0, η)
        counterfactual = findfirst(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e) ≠ factual_class, eachcol(counterfactual_candidates))
        stop -= 1
    end
    
    @info("Predicted labels: ", map(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e) ≠ factual_class, eachcol(counterfactual_candidates)))

    return counterfactual_candidates[:, counterfactual]
end
