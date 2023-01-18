using CounterfactualExplanations
using CSV
using DataFrames
using LazyArtifacts
using MLJBase
using MLJModels: ContinuousEncoder

data_dir = joinpath(artifact"data-tabular", "data-tabular")

function load_california_housing()
    df = CSV.read(joinpath(data_dir,"cal_housing.csv"), DataFrame)
    X = permutedims(Matrix(df[:,Not(:target)]))
    y = df.target
    counterfactual_data = CounterfactualData(X,y)
    return counterfactual_data
end

function load_gmsc()
    df = CSV.read(joinpath(data_dir, "gmsc.csv"), DataFrame)
    X = permutedims(Matrix(df[:, Not(:target)]))
    y = df.target
    counterfactual_data = CounterfactualData(X, y)
    return counterfactual_data
end

function load_credit_default()

    df = CSV.read(joinpath(data_dir, "credit_default.csv"), DataFrame)
    y = df.target

    # Categorical encoding:
    transformer = ContinuousEncoder()
    mach = MLJBase.fit!(machine(transformer, df[:, Not(:target)]))
    X = MLJBase.transform(mach, df[:, Not(:target)])
    X = permutedims(Matrix(X))
    features_categorical = [
        [2,3],
        collect(4:10),
        collect(11:14)
    ]

    counterfactual_data = CounterfactualData(
        X, y;
        features_categorical = features_categorical
    )

    return counterfactual_data
end