function load_mnist_mlp()
    M = Serialization.deserialize(joinpath(vision_dir, "mnist_mlp.jls"))
    return M
end

function load_mnist_ensemble()
    M = Serialization.deserialize(joinpath(vision_dir, "mnist_ensemble.jls"))
    return M
end

function load_mnist_vae(; strong=true)
    if strong
        vae = Serialization.deserialize(joinpath(vision_dir, "mnist_vae_strong.jls"))
    else
        vae = Serialization.deserialize(joinpath(vision_dir, "mnist_vae_weak.jls"))
    end
    return vae
end
