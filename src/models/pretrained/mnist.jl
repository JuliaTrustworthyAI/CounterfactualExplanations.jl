"""
    load_mnist_model(type::AbstractModelType)

Empty function to be overloaded for loading a pre-trained model for the `AbstractModelType` model type.
"""
function load_mnist_model(type::AbstractModelType) end

"""
    load_mnist_model(type::MLP)

Load a pre-trained MLP model for the MNIST dataset.
"""
function load_mnist_model(type::MLP)
    model = Serialization.deserialize(joinpath(vision_dir, "mnist_mlp.jls"))
    M = type(model; likelihood=:classification_multi)
    return M
end

"""
    load_mnist_model(type::DeepEnsemble)

Load a pre-trained deep ensemble model for the MNIST dataset.
"""
function load_mnist_model(type::DeepEnsemble)
    model = Serialization.deserialize(joinpath(vision_dir, "mnist_ensemble.jls"))
    M = type(model; likelihood=:classification_multi)
    return M
end

"""
    load_mnist_vae(; strong=true)

Load a pre-trained VAE model for the MNIST dataset.
"""
function load_mnist_vae(; strong=true)
    if strong
        vae = Serialization.deserialize(joinpath(vision_dir, "mnist_vae_strong.jls"))
    else
        vae = Serialization.deserialize(joinpath(vision_dir, "mnist_vae_weak.jls"))
    end
    return vae
end
