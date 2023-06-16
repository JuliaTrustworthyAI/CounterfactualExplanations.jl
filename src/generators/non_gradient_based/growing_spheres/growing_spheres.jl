using LinearAlgebra, Random

"Growing Spheres counterfactual generator class."
mutable struct GrowingSpheresGenerator <: AbstractNonGradientBasedGenerator
    # Add arguments
end

# TODO
# function GrowingSpheresGenerator(; )
#     return GrowingSpheresGenerator(; )
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
function hyper_sphere_coordinates(n_search_samples::Int, instance::Vector{Float64}, low::Int, high::Int; p_norm::Int=2)
    delta_instance = randn(n_search_samples, length(instance))
    dist = rand(n_search_samples) .* (high - low) .+ low  # length range [l, h)
    norm_p = LinearAlgebra.norm(delta_instance, p_norm)
    d_norm = dist ./ norm_p  # rescale/normalize factor
    delta_instance .= delta_instance .* d_norm
    instance_matrix = repeat(reshape(instance, 1, length(instance)), n_search_samples)
    candidate_counterfactuals = instance_matrix + delta_instance

    return candidate_counterfactuals, dist
end