using MLJModels: OneHotEncoder
using StatsBase: sample, Weights

@testset "Categorical data" begin

    # Set up a simple categorical dataset:
    N = 1000
    X, ys = MLJBase.make_blobs(
        N, 2; centers=2, as_table=false, center_box=(-5 => 5), cluster_std=0.5
    )
    ys .= ys .== 2
    cat_values = ["X", "Y", "Z"]
    xcat = map(ys) do y
        if y == 1
            x = sample(cat_values, Weights([0.8, 0.1, 0.1]))
        else
            x = sample(cat_values, Weights([0.1, 0.1, 0.8]))
        end
    end
    xcat = categorical(xcat)
    X = (x1=X[:, 1], x2=X[:, 2], x3=xcat)

    # Encode the categorical feature:
    hot = OneHotEncoder()
    mach = MLJBase.fit!(machine(hot, X))
    W = MLJBase.transform(mach, X)
    X = permutedims(MLJBase.matrix(W))

    # Create a counterfactual data object:
    features_categorical = [collect(3:size(X, 1))]
    counterfactual_data = CounterfactualData(
        X, ys'; features_categorical=features_categorical
    )

    # Fit a model:
    M = fit_model(counterfactual_data, :Linear)

    # Sample a factual:
    target = 1
    factual = 0
    chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
    x = select_factual(counterfactual_data, chosen)

    # Generate a counterfactual:
    generator = GenericGenerator()
    ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

    # Test:
    @test true
end
