using CategoricalArrays
using DecisionTree
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models: predict_label

"""
    grow_surrogate(
        generator::Generators.TCRExGenerator, X::AbstractArray, ŷ::AbstractArray
    )

Grows the tree-based surrogate model for the [`Generators.TCRExGenerator`](@ref). $DOC_TCREx
"""
function grow_surrogate(
    generator::Generators.TCRExGenerator, X::AbstractArray, ŷ::AbstractArray
)

    # Grow tree/forest:
    min_fraction = generator.ρ
    min_samples = round(Int, min_fraction * size(X, 1))
    if !generator.forest
        tree = MLJDecisionTreeInterface.DecisionTreeClassifier(;
            min_samples_leaf=min_samples
        )
    else
        tree = MLJDecisionTreeInterface.RandomForestClassifier(;
            min_samples_leaf=min_samples
        )
    end
    Xtab = MLJBase.table(X)
    mach = machine(tree, Xtab, ŷ) |> MLJBase.fit!

    # Return surrogate:
    return mach.model, mach.fitresult
end

"""
    grow_surrogate(
        generator::Generators.TCRExGenerator, data::CounterfactualData, M::AbstractModel
    )

Overloads the `grow_surrogate` function to accept a `CounterfactualData` and a `AbstractModel` to grow a surrogate model. See [`grow_surrogate(generator::Generators.TCRExGenerator, X::AbstractArray, ŷ::AbstractArray)`](@ref).
"""
function grow_surrogate(
    generator::Generators.TCRExGenerator, data::CounterfactualData, M::AbstractModel
)
    # Data:
    X = data.X |> permutedims                           # training samples
    ŷ = predict_label(M, data) |> categorical     # predicted outputs

    # Return surrogate:
    return grow_surrogate(generator, X, ŷ)
end

"""
    extract_rules(root::DT.Root)

Extracts decision rules (i.e. hyperrectangles) from a decision tree (`root`). For a decision tree with ``L`` leaves this results in ``2L-1`` hyperrectangles. The rules are returned as a vector of vectors of 2-element tuples, where each tuple stores the lower and upper bound imposed by the given rule for a given feature. $DOC_TCREx
"""
function extract_rules(root::DT.Root)
    conditions = [[-Inf, Inf] for i in 1:(root.n_feat)]
    conditions = vcat([conditions], extract_rules(root.node, conditions))
    conditions = [[tuple.(bounds...) for bounds in rule] for rule in conditions]
    return conditions
end

"""
    extract_rules(node::DT.Node, conditions::AbstractArray)

See [`extract_rules(root::DT.Root)`](@ref).
"""
function extract_rules(node::Union{DT.Leaf,DT.Node}, conditions::AbstractArray)
    if typeof(node) <: DT.Leaf
        # If it's a leaf node, return the accumulated conditions (a hyperrectangle)
        return []
    else
        # Get split feature and value:
        split_feature = node.featid
        threshold = node.featval

        left_conditions = deepcopy(conditions)              # left branch: feature <= threshold
        left_conditions[split_feature][2] = threshold       # upper bound
        left_hyperrectangles = extract_rules(node.left, left_conditions)

        right_conditions = deepcopy(conditions)             # right branch: feature > threshold
        right_conditions[split_feature][1] = threshold      # lower bound
        right_hyperrectangles = extract_rules(node.right, right_conditions)

        conditions = vcat(
            [left_conditions],
            left_hyperrectangles,
            [right_conditions],
            right_hyperrectangles,
        )

        return conditions
    end
end

function extract_rules(ensemble::DT.Ensemble)
    conditions = [[-Inf, Inf] for i in 1:(ensemble.n_feat)]
    conditions =
        [extract_rules(node, conditions) for node in ensemble.trees] |> x -> reduce(vcat, x) |> x -> vcat([conditions], x) |> unique
    conditions = [[tuple.(bounds...) for bounds in rule] for rule in conditions]
    return conditions
end

"""
    extract_leaf_rules(root::DT.Root)

Extracts leaf decision rules (i.e. hyperrectangles) from a decision tree (`root`). For a decision tree with ``L`` leaves this results in ``L`` hyperrectangles. The rules are returned as a vector of tuples containing 2-element tuples, where each 2-element tuple stores the lower and upper bound imposed by the given rule for a given feature. $DOC_TCREx
"""
function extract_leaf_rules(root::DT.Root)
    conditions = [[-Inf, Inf] for i in 1:(root.n_feat)]
    decisions = [nothing]
    conditions, decisions = extract_leaf_rules(root.node, conditions, decisions)
    _keep = .![isnothing.(decision)[1] for decision in decisions]
    conditions = [[tuple.(bounds...) for bounds in rule] for rule in conditions[_keep]]
    decisions = [decision[1] for decision in decisions[_keep]]
    return conditions, decisions
end

"""
    extract_leaf_rules(node::Union{DT.Leaf,DT.Node}, conditions::AbstractArray, decisions::AbstractArray)

See [`extract_leaf_rules(root::DT.Root)`](@ref) for details.
"""
function extract_leaf_rules(
    node::Union{DT.Leaf,DT.Node}, conditions::AbstractArray, decisions::AbstractArray
)
    if typeof(node) <: DT.Leaf
        # If it's a leaf node, return the accumulated conditions (a hyperrectangle)
        return [], []
    else
        left_conditions = deepcopy(conditions)              # left branch: feature <= threshold
        right_conditions = deepcopy(conditions)            # right branch: feature > threshold
        left_decisions = deepcopy(decisions)
        right_decisions = deepcopy(decisions)

        # Get split feature and value:
        split_feature = node.featid
        threshold = node.featval
        left_conditions[split_feature][2] = threshold       # upper bound
        if typeof(node.left) <: DT.Leaf
            left_decisions = [node.left.majority]
        end

        left_hyperrectangles, later_left_decisions = extract_leaf_rules(
            node.left, left_conditions, left_decisions
        )

        # Get split feature and value:
        split_feature = node.featid
        threshold = node.featval
        right_conditions[split_feature][1] = threshold        # lower bound
        if typeof(node.right) <: DT.Leaf
            right_decisions = [node.right.majority]
        end

        right_hyperrectangles, later_right_decisions = extract_leaf_rules(
            node.right, right_conditions, right_decisions
        )

        # Return the union of the two hyperrectangles:
        conditions = vcat(
            [left_conditions],
            left_hyperrectangles,
            [right_conditions],
            right_hyperrectangles,
        )

        decisions = vcat(
            [left_decisions], later_left_decisions, [right_decisions], later_right_decisions
        )

        return conditions, decisions
    end
end

include("tree.jl")

"""
    wrap_decision_tree(node::TreeNode)

See [`wrap_decision_tree(node::TreeNode, X, y)`](@ref).
"""
function wrap_decision_tree(node::TreeNode)
    if is_leaf(node)
        return DT.Leaf(node.prediction, node.values)
    else
        return DT.Node(
            node.feature,
            node.threshold,
            wrap_decision_tree(node.left),
            wrap_decision_tree(node.right),
        )
    end
end

"""
    wrap_decision_tree(node::TreeNode, X, y)

Turns a custom decision tree into a `DecisionTree.Root` object from the DecisionTree.jl package.
"""
function wrap_decision_tree(node::TreeNode, X, y, niter=3)

    # Turn into DT.Node
    node = wrap_decision_tree(node)
    X = X[:, :] |> x -> convert.(typeof(node.featval), x)
    featim = DT.permutation_importance(
        node, y, X, (model, y, X) -> DT.accuracy(y, DT.apply_tree(model, X)), niter
    )

    return DT.Root(node, length(featim.mean), featim.mean)
end

"""
    rule_feasibility(rule, X)

Computes the feasibility of a rule ``R_i`` for a given dataset. Feasibility is defined as fraction of the data points that satisfy the rule. $DOC_TCREx
"""
function rule_feasibility(rule, X)
    checks = 0
    for x in eachcol(X)
        per_feature = [lb <= x[i] < ub for (i, (lb, ub)) in enumerate(rule)]
        checks += Int(all(per_feature))
    end
    return checks / size(X, 2)
end

"""
    rule_accuracy(rule, X, fx, target)

Computes the accuracy of the rule on the data `X` for predicted outputs `fx` and the `target`. Accuracy is defined as the fraction of points contained by the rule, for which predicted values match the target. $DOC_TCREx
"""
function rule_accuracy(rule, X, fx, target)
    ingroup = 0
    checks = 0
    for (j, x) in enumerate(eachcol(X))
        # Check that sample is contained in the rule:
        per_feature = [lb <= x[i] < ub for (i, (lb, ub)) in enumerate(rule)]
        if all(per_feature)
            # Add to group:
            ingroup += 1
            # Check that 𝒻 gives an output in target:
            checks += Int(fx[j] ∈ target)
        end
    end
    return checks / ingroup
end

"""
    issubrule(rule, otherrule)

Checks if the `rule` hyperrectangle is a subset of the `otherrule` hyperrectangle. $DOC_TCREx
"""
function issubrule(rule, otherrule)
    return all([y[1] <= x[1] && x[2] <= y[2] for (x, y) in zip(rule, otherrule)])
end

"""
    max_valid(rules, X, fx, target, τ)

Returns the maximal-valid rules for a given `target` and accuracy threshold `τ`. $DOC_TCREx
"""
function max_valid(rules, X, fx, target, τ)

    # Consider only rules that meet accuracy threshold:
    candidate_rules =
        findall(rule_accuracy.(rules, (X,), (fx,), (target,)) .>= τ) |> idx -> rules[idx]

    max_valid_rules = []

    for (i, rule) in enumerate(candidate_rules)
        other_rules = candidate_rules[setdiff(eachindex(candidate_rules), i)]
        if all([!issubrule(rule, otherrule) for otherrule in other_rules])
            push!(max_valid_rules, rule)
        end
    end

    return max_valid_rules
end

"""
    partition_bounds(rules, dim::Int)

Computes the set of (unique) bounds for each rule in `rules` along the `dim`-th dimension. $DOC_TCREx
"""
function partition_bounds(rules, dim::Int)
    bounds = [-Inf, Inf]
    for rule in rules
        lb_dim = rule[dim][1]
        ub_dim = rule[dim][2]
        lb_dim ∈ bounds || push!(bounds, lb_dim)
        ub_dim ∈ bounds || push!(bounds, ub_dim)
    end
    return bounds |> sort
end

"""
    partition_bounds(rules)

Computes the set of (unique) bounds for each rule in `rules` and all dimensions. $DOC_TCREx
"""
function partition_bounds(rules)
    D = length(rules[1])
    return [partition_bounds(rules, d) for d in 1:D]
end

"""
    induced_grid(rules)

Computes the induced grid of the given rules. $DOC_TCREx.
"""
function induced_grid(rules)

    # Extract bounds for each dimension
    bounds_per_dim = partition_bounds(rules)

    # For each dimension, take consecutive pairs of bounds
    consecutive_bounds_per_dim = [
        zip(bounds[1:(end - 1)], bounds[2:end]) for bounds in bounds_per_dim
    ]

    # Cartesian product
    return Base.Iterators.product(consecutive_bounds_per_dim...) |> unique
end

"""
    rule_contains(rule, X)

Returns the subet of `X` that is contained by rule ``R_i``. $DOC_TCREx
"""
function rule_contains(rule, X)
    contained = Bool[]
    for x in eachcol(X)
        per_feature = [lb <= x[i] < ub for (i, (lb, ub)) in enumerate(rule)]
        push!(contained, Int(all(per_feature)))
    end
    return X[:, contained]
end

"""
    prototype(rule, X; pick_arbitrary::Bool=true)

Picks an arbitrary point ``x^C \\in X`` (i.e. prototype) from the subet of ``X`` that is contained by rule ``R_i``. If `pick_arbitrary` is set to false, the prototype is instead computed as the average across all samples. $DOC_TCREx
"""
function prototype(rule, X; pick_arbitrary::Bool=true)
    if pick_arbitrary
        x = rule_contains(rule, X) |> X -> X[:, rand(1:size(X, 2))]
    else
        x = rule_contains(rule, X) |> X -> mean(X; dims=2)
    end
    return x
end

"""
    rule_changes(rule, x)

Computes the number of feature changes necessary for `x` to be contained by rule ``R_i``. $DOC_TCREx
"""
function rule_changes(rule, x)
    return sum([x[i] <= lb || ub < x[i] for (i, (lb, ub)) in enumerate(rule)])
end

"""
    rule_cost(rule, x, X)

Computes the cost for ``x`` to be contained by rule ``R_i``, where cost is defined as `rule_changes(rule, x) - rule_feasibility(rule, X)`. $DOC_TCREx 
"""
function rule_cost(rule, x, X)
    return rule_changes(rule, x) - rule_feasibility(rule, X)
end

"""
    cre(rules, x, X)

Computes the counterfactual rule explanations (CRE) for a given point ``x`` and a set of ``rules``, where the ``rules`` correspond to the set of maximal-valid rules for some given target. $DOC_TCREx
"""
function cre(rules, x, X; return_index::Bool=false)
    idx = [rule_cost(rule, x, X) for rule in rules] |> argmin
    if return_index
        return idx
    else
        return rules[idx]
    end
end

"""
    classify_prototypes(prototypes, rule_assignments, bounds)

Builds the second tree model using the given `prototypes` as inputs and their corresponding `rule_assignments` as labels. Split thresholds are restricted to the `bounds`, which can be computed using [`partition_bounds(rules)`](@ref). $(DOC_TCREx)
"""
function classify_prototypes(prototypes, rule_assignments, bounds)
    # Data:
    X = prototypes
    y = rule_assignments
    @assert length(unique(y)) > 1 "Only one rule identified."

    # Grow tree/forest:
    tree = _build_tree(X, y, Inf, 0, bounds)
    tree = wrap_decision_tree(tree, X, y)

    # Return surrogate:
    return tree
end
