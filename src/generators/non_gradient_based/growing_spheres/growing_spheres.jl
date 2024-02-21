"Growing Spheres counterfactual generator class."
mutable struct GrowingSpheresGenerator <: AbstractNonGradientBasedGenerator
    n::Union{Nothing,Integer}
    η::Union{Nothing,AbstractFloat}
    latent_space::Bool
    dim_reduction::Bool
    flag::Symbol
    a₀::AbstractFloat
    a₁::AbstractFloat
end

"""
    GrowingSpheresGenerator(; n::Int=100, η::Float64=0.1, kwargs...)

Constructs a new Growing Spheres Generator object.
"""
function GrowingSpheresGenerator(;
    n::Union{Nothing,Integer}=100, η::Union{Nothing,AbstractFloat}=0.1
)
    return GrowingSpheresGenerator(n, η, false, false, :shrink, 0.0, 0.0)
end

function growing_spheres_shrink!(ce::AbstractCounterfactualExplanation)
    # Generate random points uniformly on a sphere
    counterfactual_candidates = hyper_sphere_coordinates(
        ce.generator.n, 
        ce.x, 
        0.0, 
        ce.generator.η
    )

    # Predict labels for each candidate counterfactual
    counterfactual = find_counterfactual(
        ce,
        counterfactual_candidates
    )

    if (!isnothing(counterfactual))
        ce.generator.η /= 2
    else
        # Update the boundaries of the sphere's radius for the next phase
        ce.generator.a₀, ce.generator.a₁ = ce.generator.η, 2ce.generator.η
        ce.generator.flag = :expand
    end
end

function growing_spheres_expand!(ce::AbstractCounterfactualExplanation)
    # Generate random points uniformly on a sphere
    counterfactual_candidates = hyper_sphere_coordinates(
        ce.generator.n, 
        ce.x, 
        ce.generator.a₀, 
        ce.generator.a₁
    )

    # Predict labels for each candidate counterfactual
    counterfactual = find_counterfactual(
        ce,
        counterfactual_candidates
    )

    if (isnothing(counterfactual))
        ce.generator.a₀ = ce.generator.a₁
        ce.generator.a₁ = ce.generator.a₁ + ce.generator.η
    else
        ce.x′ = counterfactual_candidates[:, counterfactual]
        ce.generator.flag = :feature_selection
    end
end

"""
    feature_selection!(ce::AbstractCounterfactualExplanation)

Perform feature selection to find the dimension with the closest (but not equal) values between the `ce.x` (factual) and `ce.s′` (counterfactual) arrays.

# Arguments
- `ce::AbstractCounterfactualExplanation`: An instance of the `AbstractCounterfactualExplanation` type representing the counterfactual explanation.

# Returns
- `nothing`

The function iteratively modifies the `ce.s′` counterfactual array by updating its elements to match the corresponding elements in the `ce.x` factual array, one dimension at a time, until the predicted label of the modified `ce.s′` matches the predicted label of the `ce.x` array.
"""
function feature_selection!(ce::AbstractCounterfactualExplanation)
    x′ = copy(ce.x′)

    i = find_closest_dimension(ce.x, x′)
    x′[i] = ce.x[i]

    if (target_probs(ce, x′) .>= ce.convergence.decision_threshold)
        ce.x′ = x′
    else
        ce.generator.flag = :converged
    end
end

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
- `n_search_samples::Int`: The number of search samples (int > 0).
- `instance::AbstractArray`: The input point array.
- `low::AbstractFloat`: The lower bound (float >= 0, l < h).
- `high::AbstractFloat`: The upper bound (float >= 0, h > l).
- `p_norm::Integer`: The norm parameter (int >= 1).

# Returns
- `candidate_counterfactuals::Array`: An array of candidate counterfactuals.
"""
function hyper_sphere_coordinates(
    n_search_samples::Integer,
    instance::AbstractArray,
    low::AbstractFloat,
    high::AbstractFloat;
    p_norm::Integer=2,
)
    delta_instance = Random.randn(n_search_samples, length(instance))
    delta_instance = convert.(eltype(instance), delta_instance)

    # length range [l, h)
    dist = Random.rand(n_search_samples) .* (high - low) .+ low
    norm_p = LinearAlgebra.norm(delta_instance, p_norm)
    # rescale/normalize factor
    d_norm = dist ./ norm_p
    delta_instance .= delta_instance .* d_norm
    instance_matrix = repeat(reshape(instance, 1, length(instance)), n_search_samples)
    candidate_counterfactuals = instance_matrix + delta_instance

    return transpose(candidate_counterfactuals)
end

"""
    find_counterfactual(model, factual_class, counterfactual_data, counterfactual_candidates)

Find the first counterfactual index by predicting labels.

# Arguments
- `model`: The fitted model used for prediction.
- `target_class`: Expected target class.
- `counterfactual_data`: Data required for counterfactual generation.
- `counterfactual_candidates`: The array of counterfactual candidates.

# Returns
- `counterfactual`: The index of the first counterfactual found.
"""
function find_counterfactual(ce, counterfactual_candidates)
    predicted_target_probabilities = map(
        e -> target_probs(ce, e)[1], eachcol(counterfactual_candidates)
    )
    predicted_counterfactual = findfirst(
        predicted_target_probabilities .>= ce.convergence.decision_threshold
    )

    return predicted_counterfactual
end

"""
    find_closest_dimension(factual, counterfactual)

Find the dimension with the closest (but not equal) values between the `factual` and `counterfactual` arrays.

# Arguments
- `factual`: The factual array.
- `counterfactual`: The counterfactual array.

# Returns
- `closest_dimension`: The index of the dimension with the closest values.

The function iterates over the indices of the `factual` array and calculates the absolute difference between the corresponding elements in the `factual` and `counterfactual` arrays. It returns the index of the dimension with the smallest difference, excluding dimensions where the values in `factual` and `counterfactual` are equal.
"""
function find_closest_dimension(factual, counterfactual)
    min_diff = typemax(eltype(factual))
    closest_dimension = -1

    for i in eachindex(factual)
        diff = abs(factual[i] - counterfactual[i])
        if diff < min_diff && factual[i] != counterfactual[i]
            min_diff = diff
            closest_dimension = i
        end
    end

    return closest_dimension
end
