function load_cifar_10_mlp()
    model = Serialization.deserialize(joinpath(vision_dir, "cifar_10_mlp.jls"))
    M = MLP(model; likelihood=:classification_multi)
    return M
end

function load_cifar_10_ensemble()
    model = Serialization.deserialize(joinpath(vision_dir, "cifar_10_ensemble.jls"))
    M = DeepEnsemble(model; likelihood=:classification_multi)
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
