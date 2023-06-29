function load_cifar_10_mlp()
    M = Serialization.deserialize(joinpath(vision_dir, "cifar_10_mlp.jls"))
    return M
end

function load_cifar_10_ensemble()
    M = Serialization.deserialize(joinpath(vision_dir, "cifar_10_ensemble.jls"))
    return M
end

function load_cifar_10_vae(; strong=true)
    if strong
        vae = Serialization.deserialize(joinpath(vision_dir, "cifar_10_vae_strong.jls"))
    else
        vae = Serialization.deserialize(joinpath(vision_dir, "cifar_10_vae_weak.jls"))
    end
    return vae
end
