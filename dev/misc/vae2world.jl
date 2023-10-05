using CounterfactualExplanations.Data: load_mnist
using CounterfactualExplanations.Models: load_mnist_vae, load_mnist_mlp
using CSV
using DataFrames
using Flux
using Images
using MLJBase
using MLJModels
using OneHotArrays
using Plots

# Load MNIST data:
data = load_mnist()
X = data.X
vae = load_mnist_vae()
mlp = load_mnist_mlp().model

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
Xtrain = MLJBase.transform(mach, X) |> 
    MLJBase.matrix |> 
    permutedims |>
    x -> Float32.(x)
# One-hot encoding:
y = fifa_world_data.y
ytrain = OneHotArrays.onehotbatch(y, 0:9)
# Dataloader:
dl = Flux.DataLoader((Xtrain, ytrain), batchsize=32, shuffle=true)
# Tokenizer:
latent = 32
activation = relu
function head(Xhat)
    return mlp(Xhat)
end
# A small MLP as our backbone, then a linear layer to map to the latent space:
tokenizer = Chain(
    Dense(size(Xtrain, 1) => latent, activation),
    Dense(latent => latent, activation),
    Dense(latent => vae.params.latent_dim),
)
# The decoder of our VAE:
reconstructor = Chain(
    vae.decoder,
    x -> clamp.(x, -1, 1),
)
# A pre-trained MLP as our head to predict labels for the generated tokens:
model = Chain(
    tokenizer,
    reconstructor,
    head,
)
loss(ŷ,y) = Flux.logitcrossentropy(ŷ, y)
opt_state = Flux.setup(Adam(), model)
# Train:
epochs = 10
for epoch in 1:epochs
    Flux.train!(model, dl, opt_state) do m, x, y
        loss(m(x), y)
    end
end

# Linear Probes 
# 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥



# Plotting:
function plot_latent(vae, X, y; n=2500)
    plt = scatter()
    μ, _ = vae.encoder(X)
    pc = svd(μ).Vt
    x1 = pc[1, :]
    x2 = pc[2, :]
    idx = !isnothing(n) ? rand(1:size(X, 2), n) : 1:size(X, 2)
    scatter!(
        x1[idx],
        x2[idx];
        markerstrokewidth=0,
        markeralpha=0.8,
        aspect_ratio=1,
        markercolor=y[idx],
        label="",
    )
    return plt
end