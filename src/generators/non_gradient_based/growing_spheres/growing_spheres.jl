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


"""
    growing_spheres_generation(generator, model, factual, counterfactual_data)

Generate counterfactual candidates using the growing spheres generation algorithm.

# Arguments
- `generator::GrowingSpheresGenerator`: An instance of the `GrowingSpheresGenerator` type representing the generator.
- `model::AbstractFittedModel`: The fitted model used for prediction.
- `factual::AbstractArray`: The factual observation to be interpreted.
- `counterfactual_data::CounterfactualData`: Data required for counterfactual generation.

# Returns
- `counterfactual_candidates`: An array of counterfactual candidates.

This function applies the growing spheres generation algorithm to generate counterfactual candidates. It starts by generating random points uniformly on a sphere, gradually reducing the search space until no counterfactuals are found. Then it expands the search space until at least one counterfactual is found or the maximum number of iterations is reached.

The algorithm iteratively generates counterfactual candidates and predicts their labels using the `model`. It checks if any of the predicted labels are different from the factual class. The process of reducing the search space involves halving the search radius, while the process of expanding the search space involves increasing the search radius.

If no counterfactual is found within the maximum number of iterations, a warning message is displayed.
"""
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

    # Predict labels for each candidate counterfactual
    predicted_labels = map(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e), eachcol(counterfactual_candidates))
    counterfactual = findfirst(predicted_labels .≠ factual_class)
    max_iteration = 1000

    # Repeat until there's no counterfactual points (process of removing all counterfactuals by reducing the search space)
    while(!isnothing(counterfactual) && max_iteration > 0)
        η = η / 2

        counterfactual_candidates = hyper_sphere_coordinates(n, factual, 0.0, η)
        predicted_labels = map(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e), eachcol(counterfactual_candidates))
        counterfactual = findfirst(predicted_labels .≠ factual_class)

        max_iteration -= 1
        if (max_iteration == 0)
            println("Warning: Maximum iteration reached. No counterfactual found.")
        end
    end
    
    # Initialize boundaries of the spehere's radius
    a₀ = η
    a₁ = 2η

    # Predict labels for each candidate counterfactual
    predicted_labels = map(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e), eachcol(counterfactual_candidates))
    counterfactual = findfirst(predicted_labels .≠ factual_class)
    max_iteration = 1000

    # Repeat until there's at least one counterfactual (process of expanding the search space)
    while(isnothing(counterfactual) && max_iteration > 0)
        a₀ = a₁
        a₁ = a₁ + η

        counterfactual_candidates = hyper_sphere_coordinates(n, factual, a₀, a₁)
        predicted_labels = map(e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e), eachcol(counterfactual_candidates))
        counterfactual = findfirst(predicted_labels .≠ factual_class)
    
        max_iteration -= 1
        if (max_iteration == 0)
            println("Warning: Maximum iteration reached. No counterfactual found.")
        end
    end

    return counterfactual_candidates[:, counterfactual]
end
