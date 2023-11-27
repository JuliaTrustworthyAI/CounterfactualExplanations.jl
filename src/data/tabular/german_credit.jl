"""
    load_german_credit(n::Union{Nothing, Int}=nothing)

Loads and pre-processes UCI German Credit data.

# Arguments
- `n::Union{Nothing, Int}=nothing`: The number of samples to subsample from the dataset. If `n` is not specified, all samples will be used. Must be <= 1000 and >= 1.

# Returns
- `counterfactual_data::CounterfactualData`: A `CounterfactualData` object containing the preprocessed data.

# Example
data = load_german_credit(500) # loads and preprocesses 500 samples from the German Credit dataset

"""
function load_german_credit(n::Union{Nothing,Int}=nothing)
    # Throw an exception if n > 1000:
    if !isnothing(n) && n > 1000
        throw(ArgumentError("n must be <= 1000"))
    end

    # Throw an exception if n < 1:
    if !isnothing(n) && n < 1
        throw(ArgumentError("n must be >= 1"))
    end

    # Load:
    df = CSV.read(joinpath(data_dir, "german_credit.csv"), DataFrames.DataFrame)

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
