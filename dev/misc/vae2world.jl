using CounterfactualExplanations.Data: load_mnist
using CounterfactualExplanations.Models: load_mnist_vae
using CSV
using DataFrames
using Flux
using MLJBase
using MLJModels
using OneHotArrays

# Load MNIST data:
data = load_mnist()
X = data.X

# World Data:
world_data = CSV.read("dev/misc/world_place.csv", DataFrame)

# FIFA World Rankings
# https://www.fifa.com/fifa-world-ranking/men?dateId=id14142
fifa_world_ranking = Dict(
    "Argentina" => 0,
    "France" => 1,
    "Brazil" => 2,
    "England" => 3,
    "Belgium" => 4,
    "Croatia" => 5,
    "Netherlands" => 6,
    "Portugal" => 7,
    "Italy" => 8,
    "Spain" => 9,
)

# Add FIFA World Rankings to World Data:
fifa_world_data = DataFrames.subset(world_data, :country => ByRow(x -> haskey(fifa_world_ranking, x))) |>
    x -> DataFrames.transform(x, :country => ByRow(x -> fifa_world_ranking[x]) => :y) |>
    x -> DataFrames.select(x, :y, Not(:y, :country))

# Tokenizer for FIFA World Data:
# 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
# Continuous feature encoding:
X = fifa_world_data[:,Not(:y)]
model = (MLJModels.ContinuousEncoder() |> MLJModels.Standardizer())
mach = machine(model, X)
MLJBase.fit!(mach)
Xtrain = MLJBase.transform(mach, X) |> MLJBase.matrix |> permutedims
# One-hot encoding:
y = fifa_world_data.y
y = OneHotArrays.onehotbatch(y, 0:9)
# Tokenizer:
tokenizer = Chain(
    Dense()
)

# Encoding:
vae = load_mnist_vae()
vae.encoder