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