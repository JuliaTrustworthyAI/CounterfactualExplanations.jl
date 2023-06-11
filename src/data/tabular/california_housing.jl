"""
    load_california_housing(n::Union{Nothing,Int}=5000)

Loads and pre-processes California Housing data.
"""
function load_california_housing(n::Union{Nothing,Int}=5000)

    # Load:
    df = read(joinpath(data_dir, "cal_housing.csv"), DataFrame)

    # Pre-process features:
    transformer = Standardizer(; count=true)
    mach = fit!(machine(transformer, df[:, Not(:target)]))
    X = transform(mach, df[:, Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = Int.(df.target)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    # Undersample:
    if !isnothing(n)
        counterfactual_data = subsample(counterfactual_data, n)
    end

    return counterfactual_data
end
