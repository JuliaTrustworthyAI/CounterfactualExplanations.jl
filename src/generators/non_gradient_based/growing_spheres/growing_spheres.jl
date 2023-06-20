"Growing Spheres counterfactual generator class."
mutable struct GrowingSpheresGenerator <: AbstractNonGradientBasedGenerator
    n::Union{Nothing,Integer}
    η::Union{Nothing,AbstractFloat}
    latent_space::Bool
end

"""
TODO: update ce accordingly (update path, convergence yada yada...)
TODO: update comments
"""

"""
    GrowingSpheresGenerator(; n::Int=100, η::Float64=0.1, kwargs...)

Constructs a new Growing Spheres Generator object.
"""
function GrowingSpheresGenerator(; 
    n::Union{Nothing,Integer}=100,
    η::Union{Nothing,AbstractFloat}=0.1,
)
    return GrowingSpheresGenerator(n, η, false)
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
function growing_spheres_generation(ce::AbstractCounterfactualExplanation)
    # Rewrite bluh bluh, easier to read
    generator = ce.generator
    model = ce.M
    factual = ce.x
    counterfactual_data = ce.data

    # Copy hyperparameters
    n = generator.n
    η = generator.η

    # Generate random points uniformly on a sphere
    counterfactual_candidates = hyper_sphere_coordinates(n, factual, 0.0, η)
    # factual_class = CounterfactualExplanations.factual_label(ce)
    factual_class = CounterfactualExplanations.Models.predict_label(
        model, counterfactual_data, factual
    )

    # Predict labels for each candidate counterfactual
    counterfactual = find_counterfactual(
        model, factual_class, counterfactual_data, counterfactual_candidates
    )
    max_iteration = 1000

    # Repeat until there's no counterfactual points (process of removing all counterfactuals by reducing the search space)
    while (!isnothing(counterfactual))
        η /= 2

        counterfactual_candidates = hyper_sphere_coordinates(n, factual, 0.0, η)
        counterfactual = find_counterfactual(
            model, factual_class, counterfactual_data, counterfactual_candidates
        )
        
        max_iteration -= 1
        if (max_iteration == 0)
            @error("Warning: Maximum iteration reached. No counterfactual found.")
        end
    end

    # Add smalles circle of candidates to path
    append!(ce.search[:path], counterfactual_candidates)

    # Initialize boundaries of the sphere's radius
    a₀, a₁ = η, 2η

    max_iteration = 1000

    # Repeat until there's at least one counterfactual (process of expanding the search space)
    while (isnothing(counterfactual))
        a₀ = a₁
        a₁ += η

        foreach(
            # candidate -> ce.search[:path] = push!(ce.search[:path], candidate),
            candidate -> ce.search[:path] = vcat(ce.search[:path], candidate),
            eachcol(counterfactual_candidates)
        )
        
        counterfactual_candidates = hyper_sphere_coordinates(n, factual, a₀, a₁)
        counterfactual = find_counterfactual(
            model, factual_class, counterfactual_data, counterfactual_candidates
        )

        max_iteration -= 1
        if (max_iteration == 0)
            @error("Warning: Maximum iteration reached. No counterfactual found.")
        end
    end

    ce.s′ = counterfactual_candidates[:, counterfactual]
    return nothing
end

"""
    feature_selection(model::AbstractFittedModel, counterfactual_data::CounterfactualData, factual::AbstractArray, counterfactual::AbstractArray)

    Perform feature selection to find the dimension with the closest (but not equal) values between the `factual` and `counterfactual` arrays.

    # Arguments
    - `model::AbstractFittedModel`: The fitted model used for prediction.
    - `counterfactual_data::CounterfactualData`: Data required for counterfactual explanation generation.
    - `factual::AbstractArray`: The factual array.
    - `counterfactual::AbstractArray`: The counterfactual array.

    # Returns
    - `counterfactual′`: The modified counterfactual array.

    The function iteratively modifies the `counterfactual` array by updating its elements to match the corresponding elements in the `factual` array, one dimension at a time, until the predicted label of the modified `counterfactual` matches the predicted label of the `factual` array.
"""
function feature_selection(ce::AbstractCounterfactualExplanation)
    model = ce.M
    counterfactual_data = ce.data
    factual = ce.x

    counterfactual′ = ce.s′
    counterfactual″ = ce.s′

    factual_class = CounterfactualExplanations.Models.predict_label(
        model, counterfactual_data, factual
    )[1]

    while (
        factual_class != CounterfactualExplanations.Models.predict_label(
            model, counterfactual_data, counterfactual′
        )
    )
        counterfactual″ = counterfactual′
        i = find_closest_dimension(factual, counterfactual′)
        counterfactual′[i] = factual[i]
    end

    ce.s′ = counterfactual″
    ce.search[:terminated] = true
    ce.search[:converged] = true
    
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
    - n_search_samples::Int64: The number of search samples (int > 0).
    - instance::Array: The Julia input point array.
    - low::Float64: The lower bound (float >= 0, l < h).
    - high::Float64: The upper bound (float >= 0, h > l).
    - p_norm::Float64: The norm parameter (float >= 1).

    # Returns
    - candidate_counterfactuals: An array of candidate counterfactuals.
    - dist: An array of distances corresponding to each candidate counterfactual.
"""
function hyper_sphere_coordinates(
    n_search_samples::Integer,
    instance::AbstractArray,
    low::AbstractFloat,
    high::AbstractFloat;
    p_norm::Integer=2,
)
    delta_instance = Random.randn(n_search_samples, length(instance))
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
    - `factual_class`: The class label of the factual observation.
    - `counterfactual_data`: Data required for counterfactual generation.
    - `counterfactual_candidates`: The array of counterfactual candidates.

    # Returns
    - `counterfactual`: The index of the first counterfactual found.
"""
function find_counterfactual(
    model, factual_class, counterfactual_data, counterfactual_candidates
)
    predicted_labels = map(
        e -> CounterfactualExplanations.Models.predict_label(model, counterfactual_data, e),
        eachcol(counterfactual_candidates),
    )
    counterfactual = findfirst(predicted_labels .≠ factual_class)

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
