"Growing Spheres counterfactual generator class."
mutable struct GrowingSpheresGenerator <: AbstractNonGradientBasedGenerator
    n::Union{Nothing,Integer}
    η::Union{Nothing,AbstractFloat}
    latent_space::Bool
    dim_reduction::Bool
end

"""
    GrowingSpheresGenerator(; n::Int=100, η::Float64=0.1, kwargs...)

Constructs a new Growing Spheres Generator object.
"""
function GrowingSpheresGenerator(;
    n::Union{Nothing,Integer}=100, η::Union{Nothing,AbstractFloat}=0.1
)
    return GrowingSpheresGenerator(n, η, false, false)
end

"""
    growing_spheres_generation(ce::AbstractCounterfactualExplanation)

Generate counterfactual candidates using the growing spheres generation algorithm.

# Arguments
- `ce::AbstractCounterfactualExplanation`: An instance of the `AbstractCounterfactualExplanation` type representing the counterfactual explanation.

# Returns
- `nothing`

This function applies the growing spheres generation algorithm to generate counterfactual candidates. It starts by generating random points uniformly on a sphere, gradually reducing the search space until no counterfactuals are found. Then it expands the search space until at least one counterfactual is found or the maximum number of iterations is reached.

The algorithm iteratively generates counterfactual candidates and predicts their labels using the model stored in `ce.M`. It checks if any of the predicted labels are different from the factual class. The process of reducing the search space involves halving the search radius, while the process of expanding the search space involves increasing the search radius.
"""
function growing_spheres_generation!(ce::AbstractCounterfactualExplanation)
    generator = ce.generator
    model = ce.M
    factual = ce.x
    counterfactual_data = ce.data
    target = [ce.target]

    # Copy hyperparameters
    n = generator.n
    η = convert(eltype(factual), generator.η)

    if (factual == target)
        ce.s′ = factual
        return nothing
    end

    # Generate random points uniformly on a sphere
    counterfactual_candidates = hyper_sphere_coordinates(n, factual, 0.0, η)

    # Predict labels for each candidate counterfactual
    counterfactual = find_counterfactual(
        model, target, counterfactual_data, counterfactual_candidates
    )

    # Repeat until there's no counterfactual points (process of removing all counterfactuals by reducing the search space)
    while (!isnothing(counterfactual) && ce.convergence[:max_iter] > 0)
        η /= 2
        a₀ = convert(eltype(factual), 0.0)

        counterfactual_candidates = hyper_sphere_coordinates(n, factual, a₀, η)
        counterfactual = find_counterfactual(
            model, target, counterfactual_data, counterfactual_candidates
        )

        ce.convergence[:max_iter] -= 1
    end

    # Update path
    ce.search[:iteration_count] += n # <- might be wrong
    for i in eachindex(counterfactual_candidates[1, :])
        push!(ce.search[:path], reshape(counterfactual_candidates[:, i], :, 1))
    end

    # Initialize boundaries of the sphere's radius
    a₀, a₁ = η, 2η

    # Repeat until there's at least one counterfactual (process of expanding the search space)
    while (isnothing(counterfactual) && ce.convergence[:max_iter] > 0)
        a₀ = a₁
        a₁ += η

        counterfactual_candidates = hyper_sphere_coordinates(n, factual, a₀, a₁)
        counterfactual = find_counterfactual(
            model, target, counterfactual_data, counterfactual_candidates
        )

        ce.convergence[:max_iter] -= 1
    end

    # Update path
    ce.search[:iteration_count] += n # Is this correct?
    for i in eachindex(counterfactual_candidates[1, :])
        push!(ce.search[:path], reshape(counterfactual_candidates[:, i], :, 1))
    end

    ce.s′ = counterfactual_candidates[:, counterfactual]
    return nothing
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
    model = ce.M
    counterfactual_data = ce.data
    factual = ce.x
    target = [ce.target]

    # Assign the initial counterfactual to both counterfactual′ and counterfactual″
    counterfactual′ = ce.s′
    counterfactual″ = ce.s′

    factual_class = CounterfactualExplanations.Models.predict_label(
        model, counterfactual_data, factual
    )[1]

    while (
        factual_class != CounterfactualExplanations.Models.predict_label(
            model, counterfactual_data, counterfactual′
        ) &&
        target == CounterfactualExplanations.Models.predict_label(
            model, counterfactual_data, counterfactual′
        ) && ce.convergence[:max_iter] > 0
    )
        counterfactual″ = counterfactual′
        i = find_closest_dimension(factual, counterfactual′)
        counterfactual′[i] = factual[i]

        ce.search[:iteration_count] += 1
        push!(ce.search[:path], reshape(counterfactual″, :, 1))

        ce.convergence[:max_iter] -= 1
    end

    ce.s′ = counterfactual″

    if (ce.convergence[:max_iter] > 0)
        ce.search[:converged] = true
    end

    ce.search[:terminated] = true

    return nothing
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
function find_counterfactual(
    model, target_class, counterfactual_data, counterfactual_candidates
)
    predicted_labels = map(
        e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e),
        eachcol(counterfactual_candidates),
    )
    counterfactual = findfirst(predicted_labels .== target_class)

    return counterfactual
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
