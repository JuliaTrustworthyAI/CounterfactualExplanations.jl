using CategoricalArrays
using DecisionTree
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models: predict_label

function Generators.grow_surrogate(
    generator::Generators.TCRExGenerator, ce::AbstractCounterfactualExplanation; kwrgs...
)
    # Data:
    X = ce.data.X |> permutedims                        # training samples
    Xtab = MLJBase.table(X)
    ŷ = predict_label(ce.M, ce.data) |> categorical     # predicted outputs

    # Grow tree/forest:
    min_fraction = generator.ρ
    min_samples = round(Int, min_fraction * size(X, 1))
    if !generator.forest
        tree = MLJDecisionTreeInterface.DecisionTreeClassifier(;
            min_samples_leaf=min_samples,
            kwrgs...
        )
    else
        tree = MLJDecisionTreeInterface.RandomForestClassifier(;
            min_samples_leaf=min_samples,
            kwrgs...
        )
    end
    mach = machine(tree, Xtab, ŷ) |> MLJBase.fit!

    # Return surrogate:
    return mach.model, mach.fitresult
end

function Generators.extract_rules(root::DT.Root)
    conditions = [[-Inf,Inf] for i in 1:root.n_feat]
    conditions = Generators.extract_rules(root.node, conditions)
    conditions = [[tuple.(bounds...) for bounds in rule] for rule in conditions]
    return conditions
end

function Generators.extract_rules(node::Union{DT.Leaf,DT.Node}, conditions::AbstractArray)

    if typeof(node) <: DT.Leaf
        # If it's a leaf node, return the accumulated conditions (a hyperrectangle)
        return []
    else
        # Get split feature and value:
        split_feature = node.featid
        threshold = node.featval

        left_conditions = deepcopy(conditions)              # left branch: feature <= threshold
        left_conditions[split_feature][2] = threshold       # upper bound
        left_hyperrectangles = Generators.extract_rules(node.left, left_conditions)

        right_conditions = deepcopy(conditions)             # right branch: feature > threshold
        right_conditions[split_feature][1] = threshold      # lower bound
        right_hyperrectangles = Generators.extract_rules(node.right, right_conditions)

        conditions = vcat([left_conditions], left_hyperrectangles, [right_conditions], right_hyperrectangles)

        return conditions
    end

end

function Generators.wrap_decision_tree(node::Generators.TreeNode)
    if Generators.is_leaf(node)
        return DT.Leaf(node.prediction, node.values)
    else
        return DT.Node(node.feature, node.threshold, Generators.wrap_decision_tree(node.left), Generators.wrap_decision_tree(node.right))
    end
end

function Generators.wrap_decision_tree(node::Generators.TreeNode, X, y, niter=3)

    # Turn into DT.Node
    node = Generators.wrap_decision_tree(node)
    X = X[:, :] |> x -> convert.(typeof(node.featval), x)
    featim = DT.permutation_importance(
        node, y, X, (model, y, X) -> DT.accuracy(y, DT.apply_tree(model, X)), niter
    )
    
    return DT.Root(node, length(featim.mean), featim.mean)
end

"""
    Generators.classify_prototypes(prototypes, rule_assignments, bounds)

Builds the second tree model using the given `prototypes` as inputs and their corresponding `rule_assignments` as labels. Split thresholds are restricted to the `bounds`, which can be computed using [`partition_bounds(rules)`](@ref). $(Generators.DOC_TCREx)
"""
function Generators.classify_prototypes(prototypes, rule_assignments, bounds)
    # Data:
    X = prototypes
    y = rule_assignments

    # Grow tree/forest:
    tree =
        Generators._build_tree(X, y, Inf, 0, bounds) |>
        tree -> Generators.wrap_decision_tree(tree, X, y)

    # Return surrogate:
    return tree
end