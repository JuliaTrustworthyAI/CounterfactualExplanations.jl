using BSON
using CounterfactualExplanations 
using Flux
using LazyArtifacts
using Random

"""
    cats_dogs_data()

A convenience function to load the cats and dogs toy dataset.
"""
# Cats and dogs:
function cats_dogs_data()
    data_dir = artifact"cats_dogs_data"
    data = BSON.load(joinpath(data_dir, "cats_dogs_data.bson"), @__MODULE__)[:data]
    X, y = (data[:X], data[:y])
    return X, y
end

"""
    cats_dogs_model()

A convenience function to load the pre-trained MLP to classify cats and dogs.
"""
function cats_dogs_model()
    data_dir = artifact"cats_dogs_model"
    model = BSON.load(joinpath(data_dir, "cats_dogs_model.bson"), @__MODULE__)[:model]
    return model
end

"""
    cats_dogs_laplace()

A convenience function to load the pre-trained MLP with Laplace approximation to classify cats and dogs.
"""
function cats_dogs_laplace()
    data_dir = artifact"cats_dogs_laplace"
    la = BSON.load(joinpath(data_dir, "cats_dogs_laplace.bson"), @__MODULE__)[:la]
    return la
end

"""
    toy_data_linear(N=100)

A convenience function to load linearly separable synthetic data.

# Examples

```julia-repl
toy_data_linear()
```

"""
function toy_data_linear(N = 100, p = 2)
    # Number of points to generate.
    M = round(Int, N / 2)

    # Generate artificial data.
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
    xt0s = Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M])

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(M); zeros(M)]
    return xs, ts
end

"""
    toy_data_non_linear(N=100)

A convenience function to load synthetic data that are not linearly separable.

# Examples

```julia-repl
toy_data_non_linear()
```

"""
function toy_data_non_linear(N = 100)
    # Number of points to generate.
    M = round(Int, N / 4)

    # Generate artificial data.
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))

    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(2 * M); zeros(2 * M)]
    return xs, ts
end

"""
    toy_data_multi(N=100)

A convenience function to load multi-class synthetic data.

# Examples

```julia-repl
toy_data_multi()
```

"""
function toy_data_multi(N = 100)
    # Number of points to generate.
    M = round(Int, N / 4)

    # Generate artificial data.
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt1s = Array([[x1s[i] + 1; x2s[i] + 1] for i = 1:M])
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    append!(xt1s, Array([[x1s[i] - 7; x2s[i] - 7] for i = 1:M]))

    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    xt0s = Array([[x1s[i] + 1; x2s[i] - 7] for i = 1:M])
    x1s = rand(M) * 4.5
    x2s = rand(M) * 4.5
    append!(xt0s, Array([[x1s[i] - 7; x2s[i] + 1] for i = 1:M]))

    # Store all the data for later.
    xs = [xt1s; xt0s]
    ts = [ones(M); ones(M) .* 2; ones(M) .* 3; ones(M) .* 4]
    return xs, ts
end

###########################
# Synthetic data
###########################
"""
    load_synthetic()

Helper function that loads dictionary of pretrained models.
"""
function load_synthetic(models = [:flux])

    synthetic_dict = Dict()
    # Data
    data_dir = artifact"synthetic_data"
    data_dict =
        BSON.load(joinpath(data_dir, "synthetic_data.bson"), @__MODULE__)[:data_dict]
    for (likelihood, data) ∈ data_dict
        synthetic_dict[likelihood] = Dict()
        synthetic_dict[likelihood][:data] = data
        synthetic_dict[likelihood][:models] = Dict()
    end
    # Flux
    if :flux ∈ models
        data_dir = artifact"synthetic_flux"
        models_ =
            BSON.load(joinpath(data_dir, "synthetic_flux.bson"), @__MODULE__)[:flux_models]
        for (likelihood, model) ∈ models_
            synthetic_dict[likelihood][:models][:flux] = model
            synthetic_dict[likelihood][:models][:flux][:model] =
                Models.FluxModel(model[:raw_model], likelihood = likelihood)
        end
    end

    return synthetic_dict
end
