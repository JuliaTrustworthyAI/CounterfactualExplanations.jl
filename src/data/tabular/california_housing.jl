"""
    load_california_housing(n::Union{Nothing,Int}=5000)

Loads and pre-processes California Housing data.
"""
function load_california_housing(n::Union{Nothing,Int}=5000)

    # check that n is > 0
    if !isnothing(n) && n <= 0
        throw(ArgumentError("n must be > 0"))
    end

    # Load:
    df = CSV.read(joinpath(data_dir, "cal_housing.csv"), DataFrames.DataFrame)
    # Pre-process features:
    transformer = MLJModels.Standardizer(; count=true)
    mach = MLJBase.fit!(MLJBase.machine(transformer, df[:, DataFrames.Not(:target)]))
    X = MLJBase.transform(mach, df[:, DataFrames.Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = Int.(df.target)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)

    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
    end
    return counterfactual_data
end
