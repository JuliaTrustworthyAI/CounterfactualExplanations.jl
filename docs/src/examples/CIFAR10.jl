using Statistics, BSON
using Flux, Flux.Optimise
using MLDatasets: CIFAR10
using Images.ImageCore
using Flux: onehotbatch, onecold
using Base.Iterators: partition
using CUDA

# Data
train_x, train_y = CIFAR10.traindata(Float32)
labels = onehotbatch(train_y, 0:9)
N = 50000
batch_size = 1000
train = ([(train_x[:,:,:,i], labels[:,i]) for i in partition(1:(N-batch_size), batch_size)]) |> gpu
valset = (N-batch_size+1):N
valX = train_x[:,:,:,valset] |> gpu
valY = labels[:, valset] |> gpu

# # CNN
# m = Chain(
#   Conv((5,5), 3=>16, relu),
#   MaxPool((2,2)),
#   Conv((5,5), 16=>8, relu),
#   MaxPool((2,2)),
#   x -> reshape(x, :, size(x, 4)),
#   Dense(200, 120),
#   Dense(120, 84),
#   Dense(84, 10),
#   softmax) |> gpu

# using Flux: crossentropy, Momentum
# loss(x, y) = sum(crossentropy(m(x), y))
# opt = Momentum(0.01)
# accuracy(x, y) = mean(onecold(m(x), 0:9) .== onecold(y, 0:9))

# # Training
# epochs = 10
# for epoch = 1:epochs
#   for d in train
#     gs = gradient(params(m)) do
#       l = loss(d...)
#     end
#     update!(opt, params(m), gs)
#   end
#   @show accuracy(valX, valY)
# end

# BSON.@save "CIFAR10_nn.bson" nn

# Load model
BSON.@load "CIFAR10_nn.bson" nn