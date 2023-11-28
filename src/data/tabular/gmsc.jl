"""
    load_gmsc(n::Union{Nothing,Int}=5000)

Loads and pre-processes Give Me Some Credit (GMSC) data.
"""
function load_gmsc(n::Union{Nothing,Int}=5000)

    # Load:
    df = CSV.read(joinpath(data_dir, "gmsc.csv"), DataFrames.DataFrame)

    # Pre-process features:
    transformer = MLJModels.Standardizer(; count=true)
    mach = MLJBase.fit!(MLJBase.machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = df.target
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)

    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
    end

    return counterfactual_data
end
