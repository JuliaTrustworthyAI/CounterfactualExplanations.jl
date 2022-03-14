using Pkg.Artifacts
using Flux
using BSON

# UCR data:
function ucr_data()
    data_dir = artifact"ucr_data"
    data = BSON.load(joinpath(data_dir,"ucr_data.bson"),@__MODULE__)
    return data
end

function ucr_model()
    data_dir = artifact"ucr_model"
    model = BSON.load(joinpath(data_dir,"ucr_model.bson"),@__MODULE__)
    return model
end

# Cats and dogs:
function cats_dogs_data()
    data_dir = artifact"cats_dogs_data"
    data = BSON.load(joinpath(data_dir,"cats_dogs_data.bson"),@__MODULE__)[:data]
    return data
end

function cats_dogs_model()
    data_dir = artifact"cats_dogs_model"
    model = BSON.load(joinpath(data_dir,"cats_dogs_model.bson"),@__MODULE__)[:model]
    return model
end

function cats_dogs_lapalce()
    data_dir = artifact"cats_dogs_data"
    la = BSON.load(joinpath(data_dir,"cats_dogs_laplace.bson"),@__MODULE__)[:la]
    return la
end

# MNIST:
function mnist_data()
    data_dir = artifact"mnist_data"
    data = BSON.load(joinpath(data_dir,"mnist_data.bson"),@__MODULE__)[:data]
    return data
end

function mnist_model()
    data_dir = artifact"mnist_model"
    model = BSON.load(joinpath(data_dir,"mnist_model.bson"),@__MODULE__)[:model]
    return testmode!(model)
end

function mnist_ensemble()
    data_dir = joinpath(artifact"mnist_ensemble","mnist_ensemble")
    model_files = Base.Filesystem.readdir(data_dir)
    𝓜 = []
    for file in model_files
        model = BSON.load(joinpath(data_dir,file),@__MODULE__)[:model]
        𝓜 = vcat(𝓜, testmode!(model))
    end
    return 𝓜
end

using Random
"""
    toy_data_linear(N=100)

# Examples

```julia-repl
toy_data_linear()
```

"""
function toy_data_linear(N=100)
    # Number of points to generate.
    M = round(Int, N / 2)
    Random.seed!(1234)

    # Generate artificial data.
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
    xt0s = Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M])

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(M); zeros(M)];
    return xs, ts
end

using Random
"""
    toy_data_non_linear(N=100)

# Examples

```julia-repl
toy_data_non_linear()
```

"""
function toy_data_non_linear(N=100)
    # Number of points to generate.
    M = round(Int, N / 4)
    Random.seed!(1234)

    # Generate artificial data.
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))

    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(2*M); zeros(2*M)];
    return xs, ts
end

using Random
"""
    toy_data_multi(N=100)

# Examples

```julia-repl
toy_data_multi()
```

"""
function toy_data_multi(N=100)
    # Number of points to generate.
    M = round(Int, N / 4)
    Random.seed!(1234)

    # Generate artificial data.
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt1s = Array([[x1s[i] + 1; x2s[i] + 1] for i = 1:M])
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    append!(xt1s, Array([[x1s[i] - 7; x2s[i] - 7] for i = 1:M]))

    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    xt0s = Array([[x1s[i] + 1; x2s[i] - 7] for i = 1:M])
    x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
    append!(xt0s, Array([[x1s[i] - 7; x2s[i] + 1] for i = 1:M]))

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(M); ones(M).*2; ones(M).*3; ones(M).*4];
    return xs, ts
end