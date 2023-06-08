"""
    load_gmsc(n::Union{Nothing,Int}=5000)

Loads and pre-processes Give Me Some Credit (GMSC) data.
"""
function load_gmsc(n::Union{Nothing,Int}=5000)

    # Load:
    df = read(joinpath(data_dir, "gmsc.csv"), DataFrame)

    # Pre-process features:
    transformer = Standardizer(; count=true)
    mach = fit!(machine(transformer, df[:, Not(:target)]))
    X = transform(mach, df[:, Not(:target)])
    X = Matrix(X)
    X = permutedims(X)

    # Counterfactual data:
    y = df.target
    counterfactual_data = CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    # Undersample:
    if !isnothing(n)
        counterfactual_data = subsample(counterfactual_data, n)
    end

    return counterfactual_data
end
