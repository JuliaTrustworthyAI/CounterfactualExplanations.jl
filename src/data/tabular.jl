using CounterfactualExplanations
using CSV
using DataFrames
using LazyArtifacts
using MLJBase

data_dir = joinpath(artifact"data-tabular", "data-tabular")

function load_california_housing()
    df = CSV.read(joinpath(data_dir,"cal_housing.csv"), DataFrame)
    return df
end

function load_credit_default()
    df = CSV.read(joinpath(data_dir, "credit_default.csv"), DataFrame)
    return df
end

function load_gmsc()
    df = CSV.read(joinpath(data_dir, "gmsc.csv"), DataFrame)
    return df
end