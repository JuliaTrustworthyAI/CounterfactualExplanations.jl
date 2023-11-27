"""
    load_credit_default(n::Union{Nothing,Int}=5000)

Loads and pre-processes UCI Credit Default data.
"""
function load_credit_default(n::Union{Nothing,Int}=5000)

    # Load:
    df = CSV.read(joinpath(data_dir, "credit_default.csv"), DataFrames.DataFrame)

    # Pre-process features:
    df.SEX = MLJBase.categorical(df.SEX)
    df.EDUCATION = MLJBase.categorical(df.EDUCATION)
    df.MARRIAGE = MLJBase.categorical(df.MARRIAGE)
    transformer = MLJModels.Standardizer(; count=true) |> MLJModels.ContinuousEncoder()
    mach = MLJBase.fit!(MLJBase.machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])
    X = permutedims(Matrix(X))
    features_categorical = [
        [2, 3],             # SEX
        collect(4:10),      # EDUCATION
        collect(11:14),      # MARRIAGE
    ]

    # Counterfactual data:
    y = df.target
    counterfactual_data = CounterfactualExplanations.CounterfactualData(
        X, y; features_categorical=features_categorical
    )

    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
    end

    return counterfactual_data
end
