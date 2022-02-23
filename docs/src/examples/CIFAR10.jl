using Statistics, BSON
using Flux, Flux.Optimise
using MLDatasets: CIFAR10
using Images.ImageCore
using Flux: onehotbatch, onecold
using Base.Iterators: partition
using CUDA

# Data
x, y = CIFAR10.traindata(Float32)

# Turn into binary problem (cats/dogs)
class_names = CIFAR10.classnames()
n_classes = length(class_names)
c_cat = findall(class_names.=="cat") .- 1
c_dog = findall(class_names.=="dog") .- 1
cats_and_dogs = findall(y .∈ Ref(vcat(c_cat, c_dog)))

x = x[:,:,:,cats_and_dogs]
y = y[cats_and_dogs]
y = ifelse.(y.==c_cat,0,1)

n = length(y)
batch_size = Int(n/10)
train_set = 1:(n-batch_size)
val_set = (n-batch_size+1):n
train_x = x[:,:,:,train_set]
train_y = y[train_set]
val_x = x[:,:,:,val_set] |> gpu
val_y = y[val_set] |> gpu
data = Flux.DataLoader((data=train_x, label=train_y), batchsize=batch_size) |> gpu

if retrain 
    # CNN
    nn = Chain(
        Conv((5,5), 3=>16, relu),
        MaxPool((2,2)),
        Conv((5,5), 16=>8, relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(200, 120),
        Dense(120, 84),
        Dense(84, 1)
    ) |> gpu

    using Flux: Momentum
    using Flux.Losses: logitbinarycrossentropy
    loss(x, y) = logitbinarycrossentropy(vec(nn(x)), y)
    opt = Momentum(0.01)
    accuracy(x, y) = mean(vec(round.(Flux.σ.(nn(x)))) .== y)
    avg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))

    # Training
    epochs = 200
    for epoch = 1:epochs
    for d in data
        gs = gradient(params(nn)) do
        l = loss(d...)
        end
        update!(opt, params(nn), gs)
    end
    println("Training loss")
    @show avg_loss(data)
    println("Training accuracy")
    @show accuracy(train_x,train_y)
    println("Test accuracy")
    @show accuracy(val_x, val_y)
    end

    BSON.@save "CIFAR10_nn.bson" nn
end

# Load model
BSON.@load "CIFAR10_nn.bson" nn