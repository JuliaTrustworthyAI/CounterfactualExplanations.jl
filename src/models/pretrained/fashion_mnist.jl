using Serialization: Serialization

function load_fashion_mnist_mlp()
    model = Serialization.deserialize(joinpath(vision_dir, "fashion_mnist_mlp.jls"))
    M = MLP(model; likelihood=:classification_multi)
    return M
end

function load_fashion_mnist_ensemble()
    model = Serialization.deserialize(joinpath(vision_dir, "fashion_mnist_ensemble.jls"))
    M = DeepEnsemble(model; likelihood=:classification_multi)
    return M
end

function load_fashion_mnist_vae(; strong=true)
    if strong
        vae = Serialization.deserialize(
            joinpath(vision_dir, "fashion_mnist_vae_strong.jls")
        )
    else
        vae = Serialization.deserialize(joinpath(vision_dir, "fashion_mnist_vae_weak.jls"))
    end
    return vae
end
