using CategoricalArrays
using DecisionTree
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models: predict_label

function Generators.grow_surrogate(
    generator::Generators.TCRExGenerator, ce::AbstractCounterfactualExplanation
)
    # Data:
    X = ce.data.X |> permutedims                        # training samples
    Xtab = MLJBase.table(X)
    ŷ = predict_label(ce.M, ce.data) |> categorical     # predicted outputs

    # Grow tree/forest:
    min_fraction = generator.ρ
    min_samples = round(Int, min_fraction * size(X, 2))
    if !generator.forest
        tree = MLJDecisionTreeInterface.DecisionTreeClassifier(;
            min_samples_split=min_samples
        )
    else
        tree = MLJDecisionTreeInterface.RandomForestClassifier(;
            min_samples_split=min_samples
        )
    end
    mach = machine(tree, Xtab, ŷ) |> MLJBase.fit!

    # Return surrogate:
    return mach.model, mach.fitresult
end

function Generators.extract_rules(
    generator::Generators.TCRExGenerator, ce::AbstractCounterfactualExplanation
)
end