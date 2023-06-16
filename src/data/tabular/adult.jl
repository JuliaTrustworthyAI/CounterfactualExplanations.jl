"""
    load_uci_adult(n::Union{Nothing, Int}=1000)

Load and preprocesses data from the UCI 'Adult' dataset

# Arguments
- `n::Union{Nothing, Int}=nothing`: The number of samples to subsample from the dataset.

# Returns
- `counterfactual_data::CounterfactualData`: A `CounterfactualData` object containing the preprocessed data.

# Example
data = load_german_credit(20) # loads and preprocesses 20 samples from the Adult dataset
"""
function load_uci_adult(n::Union{Nothing,Int}=1000)
    # Throw an exception if n > 1000:
    if !isnothing(n) && n > 1000
        throw(ArgumentError("n must be <= 1000"))
    end
    # Throw an exception if n < 1:
    if !isnothing(n) && n < 1
        throw(ArgumentError("n must be >= 1"))
    end

    # Load data
    df = CSV.read(joinpath(data_dir, "adult.csv"), DataFrame)
    rename!(
        df,
        [
            :age,
            :workclass,
            :fnlwgt,
            :education,
            :education_num,
            :marital_status,
            :occupation,
            :relationship,
            :race,
            :sex,
            :capital_gain,
            :capital_loss,
            :hours_per_week,
            :native_country,
            :target,
        ],
    )

    # Preprocessing
    transformer = Standardizer(; count=true)
    mach = MLJBase.fit!(machine(transformer, df[:, Not(:target)]))
    X = MLJBase.transform(mach, df[:, Not(:target)])
    X = Matrix(X)
    X = permutedims(X)
    println(typeof(X))
    X = Float32.(X)

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