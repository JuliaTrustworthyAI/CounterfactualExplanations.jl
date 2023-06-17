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

"""
    growing_spheres_search(
        instance,
        keys_mutable,
        keys_immutable,
        continuous_cols,
        binary_cols,
        feature_order,
        model,
        n_search_samples=1000,
        p_norm=2,
        step=0.2,
        max_iter=1000,
    )

# Arguments
- instance: df
- step: float > 0; step_size for growing spheres
- n_search_samples: int > 0
- model: sklearn classifier object
- p_norm: float=>1; denotes the norm (classical: 1 or 2)
- max_iter: int > 0; maximum # iterations
- keys_mutable: list; list of input names we can search over
- keys_immutable: list; list of input names that may not be searched over

# Return
""" 
function growing_spheres_search(
    instance,
    keys_mutable,
    keys_immutable,
    continuous_cols,
    binary_cols,
    feature_order,
    model,
    n_search_samples=1000,
    p_norm=2,
    step=0.2,
    max_iter=1000,
)
    # correct order of names
    keys_correct = feature_order
    # divide up keys
    keys_mutable_continuous = list(set(keys_mutable) - set(binary_cols))
    keys_mutable_binary = list(set(keys_mutable) - set(continuous_cols))

    # Divide data in 'mutable' and 'non-mutable'
    # In particular, divide data in 'mutable & binary' and 'mutable and continuous'
    # instance_immutable_replicated = np.repeat(
    #     instance[keys_immutable].values.reshape(1, -1), n_search_samples, axis=0
    # )
    # instance_replicated = np.repeat(
    #     instance.values.reshape(1, -1), n_search_samples, axis=0
    # )
    # instance_mutable_replicated_continuous = np.repeat(
    #     instance[keys_mutable_continuous].values.reshape(1, -1),
    #     n_search_samples,
    #     axis=0,
    # )
    # instance_mutable_replicated_binary = np.repeat(
    #     instance[keys_mutable_binary].values.reshape(1, -1), n_search_samples, axis=0
    # )

    # # init step size for growing the sphere
    # low = 0
    # high = low + step

    # # counter
    # count = 0
    # counter_step = 1

    # # get predicted label of instance
    # instance_label = np.argmax(model.predict_proba(instance.values.reshape(1, -1)))

    # counterfactuals_found = False
    # candidate_counterfactual_star = np.empty(
    #     instance_replicated.shape[1],
    # )
    # candidate_counterfactual_star[:] = np.nan
    # while (not counterfactuals_found) and (count < max_iter):
    #     count = count + counter_step

    #     # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
    #     candidate_counterfactuals_continuous, _ = hyper_sphere_coordindates(
    #         n_search_samples, instance_mutable_replicated_continuous, high, low, p_norm
    #     )

    #     # sample random points from Bernoulli distribution
    #     candidate_counterfactuals_binary = np.random.binomial(
    #         n=1, p=0.5, size=n_search_samples * len(keys_mutable_binary)
    #     ).reshape(n_search_samples, -1)

    #     # make sure inputs are in correct order
    #     candidate_counterfactuals = pd.DataFrame(
    #         np.c_[
    #             instance_immutable_replicated,
    #             candidate_counterfactuals_continuous,
    #             candidate_counterfactuals_binary,
    #         ]
    #     )
    #     candidate_counterfactuals.columns = (
    #         keys_immutable + keys_mutable_continuous + keys_mutable_binary
    #     )
    #     # enforce correct order
    #     candidate_counterfactuals = candidate_counterfactuals[keys_correct]

    #     # STEP 2 -- COMPUTE l_1 DISTANCES
    #     if p_norm == 1:
    #         distances = np.abs(
    #             (candidate_counterfactuals.values - instance_replicated)
    #         ).sum(axis=1)
    #     elif p_norm == 2:
    #         distances = np.square(
    #             (candidate_counterfactuals.values - instance_replicated)
    #         ).sum(axis=1)
    #     else:
    #         raise ValueError("Distance not defined yet")

    #     # counterfactual labels
    #     y_candidate_logits = model.predict_proba(candidate_counterfactuals.values)
    #     y_candidate = np.argmax(y_candidate_logits, axis=1)
    #     indeces = np.where(y_candidate != instance_label)
    #     candidate_counterfactuals = candidate_counterfactuals.values[indeces]
    #     candidate_dist = distances[indeces]

    #     if len(candidate_dist) > 0:  # certain candidates generated
    #         min_index = np.argmin(candidate_dist)
    #         candidate_counterfactual_star = candidate_counterfactuals[min_index]
    #         counterfactuals_found = True

    #     # no candidate found & push search range outside
    #     low = high
    #     high = low + step

    # return candidate_counterfactual_star
end