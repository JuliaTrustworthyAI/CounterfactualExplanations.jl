using CounterfactualExplanations
using CSV
using DataFrames
using LazyArtifacts
using MLJBase
using MLJModels: ContinuousEncoder, OneHotEncoder, Standardizer

data_dir = joinpath(artifact"data-tabular", "data-tabular")

"""
    load_california_housing(n::Union{Nothing,Int}=5000)

Loads and pre-processes California Housing data.
"""
function load_california_housing(n::Union{Nothing,Int}=5000)

    # Load:
    df = CSV.read(joinpath(data_dir, "cal_housing.csv"), DataFrame)

    # Pre-process features:
    transformer = Standardizer(; count=true)
    mach = MLJBase.fit!(machine(transformer, df[:, Not(:target)]))
    X = MLJBase.transform(mach, df[:, Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = Int.(df.target)
    counterfactual_data = CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
    end

    return counterfactual_data
end

"""
    load_gmsc(n::Union{Nothing,Int}=5000)

Loads and pre-processes Give Me Some Credit (GMSC) data.
"""
function load_gmsc(n::Union{Nothing,Int}=5000)

    # Load:
    df = CSV.read(joinpath(data_dir, "gmsc.csv"), DataFrame)

    # Pre-process features:
    transformer = Standardizer(; count=true)
    mach = MLJBase.fit!(machine(transformer, df[:, Not(:target)]))
    X = MLJBase.transform(mach, df[:, Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = df.target
    counterfactual_data = CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
    end

    return counterfactual_data
end

"""
    load_credit_default(n::Union{Nothing,Int}=5000)

Loads and pre-processes UCI Credit Default data.
"""
function load_credit_default(n::Union{Nothing,Int}=5000)

    # Load:
    df = CSV.read(joinpath(data_dir, "credit_default.csv"), DataFrame)

    # Pre-process features:
    df.SEX = categorical(df.SEX)
    df.EDUCATION = categorical(df.EDUCATION)
    df.MARRIAGE = categorical(df.MARRIAGE)
    transformer = Standardizer(; count=true) |> ContinuousEncoder()
    mach = MLJBase.fit!(machine(transformer, df[:, Not(:target)]))
    X = MLJBase.transform(mach, df[:, Not(:target)])
    X = permutedims(Matrix(X))
    features_categorical = [
        [2, 3],             # SEX
        collect(4:10),      # EDUCATION
        collect(11:14),      # MARRIAGE
    ]

    # Counterfactual data:
    y = df.target
    counterfactual_data = CounterfactualData(
        X, y; features_categorical=features_categorical
    )
    counterfactual_data.X = Float32.(counterfactual_data.X)

    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
    end

    return counterfactual_data
end

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
    # Throw an exceptoin if n > 1000:
    if !isnothing(n) && n > 1000
        throw(ArgumentError("n must be <= 1000"))
    end

    # Throw an exceptoin if n < 1:
    if !isnothing(n) && n < 1
        throw(ArgumentError("n must be >= 1"))
    end

    # Load:
    df = CSV.read(joinpath(data_dir, "credit_default.csv"), DataFrame)

    # Counterfactual data:
    X = df[:, Not(:target)]
    y = df.target
    counterfactual_data = CounterfactualData(X, y)

    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
    end

    return counterfactual_data
end
