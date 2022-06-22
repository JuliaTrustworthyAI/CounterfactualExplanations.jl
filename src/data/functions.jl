using Pkg.Artifacts
using Flux
using BSON

"""
    cats_dogs_data()

A convenience function to load the cats and dogs toy dataset.
"""
# Cats and dogs:
function cats_dogs_data()
    data_dir = artifact"cats_dogs_data"
    data = BSON.load(joinpath(data_dir,"cats_dogs_data.bson"),@__MODULE__)[:data]
    X, y = (data[:X], data[:y])
    return X, y
end

"""
    cats_dogs_model()

A convenience function to load the pre-trained MLP to classify cats and dogs.
"""
function cats_dogs_model()
    data_dir = artifact"cats_dogs_model"
    model = BSON.load(joinpath(data_dir,"cats_dogs_model.bson"),@__MODULE__)[:model]
    return model
end

"""
    cats_dogs_laplace()

A convenience function to load the pre-trained MLP with Laplace approximation to classify cats and dogs.
"""
function cats_dogs_laplace()
    data_dir = artifact"cats_dogs_laplace"
    la = BSON.load(joinpath(data_dir,"cats_dogs_laplace.bson"),@__MODULE__)[:la]
    return la
end

# MNIST:
"""
    mnist_data()

A convenience function to load MNIST training data.
"""
function mnist_data()
    data_dir = artifact"mnist_data"
    data = BSON.load(joinpath(data_dir,"mnist_data.bson"),@__MODULE__)[:data]
    X, ys = (data[:X], data[:ys])
    return X, ys
end

"""
    mnist_model()

A convenience function to load the pre-trained MLP for MNIST training data.
"""
function mnist_model()
    data_dir = artifact"mnist_model"
    model = BSON.load(joinpath(data_dir,"mnist_model.bson"),@__MODULE__)[:model]
    return testmode!(model)
end

"""
    mnist_ensemble()

A convenience function to load the pre-trained deep ensemble of MLPs for MNIST training data.
"""
function mnist_ensemble()
    data_dir = joinpath(artifact"mnist_ensemble","mnist_ensemble")
    model_files = Base.Filesystem.readdir(data_dir)
    ensemble = []
    for file in model_files
        model = BSON.load(joinpath(data_dir,file),@__MODULE__)[:model]
        ensemble = vcat(ensemble, testmode!(model))
    end
    return ensemble
end

using Random
"""
    toy_data_linear(N=100)

A convenience function to load linearly separable synthetic data.

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

A convenience function to load synthetic data that are not linearly separable.

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

A convenience function to load multi-class synthetic data.

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

###########################
# Synthetic data
###########################
using Flux, RCall
"""
    load_synthetic()

Helper function that loads dictionary of pretrained models.
"""
function load_synthetic(models=[:flux, :r_torch])
    synthetic_dict = Dict()
    # Data
    data_dir = artifact"synthetic_data"
    data_dict = BSON.load(joinpath(data_dir,"synthetic_data.bson"),@__MODULE__)[:data_dict]
    for (model_type, data) ∈ data_dict
        synthetic_dict[model_type] = Dict()
        synthetic_dict[model_type][:data] = data
        synthetic_dict[model_type][:models] = Dict()
    end
    # Flux
    if :flux ∈ models
        data_dir = artifact"synthetic_flux"
        models_ = BSON.load(joinpath(data_dir,"synthetic_flux.bson"),@__MODULE__)[:flux_models]
        for (model_type, model) ∈ models_
            synthetic_dict[model_type][:models][:flux] = model
            synthetic_dict[model_type][:models][:flux][:model] = Models.FluxModel(model[:raw_model],type=model_type)
        end
    end
    # R torch
    if :r_torch ∈ models
        R"""
        library(torch)
        """
        data_dir = artifact"synthetic_r_torch"
        data_dir = joinpath(data_dir,"synthetic_r_torch")
        model_names = readdir(data_dir)
        model_paths = map(sub_dir -> joinpath(data_dir,sub_dir,"model.pt"),model_names)
        model_info = zip(model_names, model_paths)
        for (name,path) ∈ model_info
            model_type = Symbol(name)
            synthetic_dict[model_type][:models][:r_torch] = Dict()
            M = R"torch_load($path)"
            synthetic_dict[model_type][:models][:r_torch][:raw_model] = M
            synthetic_dict[model_type][:models][:r_torch][:model] = Models.RTorchModel(M, type=model_type)
        end
    end
    return synthetic_dict
end

