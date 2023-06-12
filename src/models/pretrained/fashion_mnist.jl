function load_fashion_mnist_mlp()
    M = Serialization.deserialize(joinpath(vision_dir, "fashion_mnist_mlp.jls"))
    return M
end

function load_fashion_mnist_ensemble()
    M = Serialization.deserialize(joinpath(vision_dir, "fashion_mnist_ensemble.jls"))
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
