using LazyArtifacts
using Serialization

vision_dir = CounterfactualExplanations.generate_artifact_dir("model-vision")

function load_mnist_mlp()
    M = deserialize(joinpath(vision_dir, "mnist_mlp.jls"))
    return M
end

function load_mnist_ensemble()
    M = deserialize(joinpath(vision_dir, "mnist_ensemble.jls"))
    return M
end

function load_mnist_vae(; strong=true)
    if strong
        vae = deserialize(joinpath(vision_dir, "mnist_vae_strong.jls"))
    else
        vae = deserialize(joinpath(vision_dir, "mnist_vae_weak.jls"))
    end
    return vae
end

function load_fashion_mnist_mlp()
    M = deserialize(joinpath(vision_dir, "fashion_mnist_mlp.jls"))
    return M
end

function load_fashion_mnist_ensemble()
    M = deserialize(joinpath(vision_dir, "fashion_mnist_ensemble.jls"))
    return M
end

function load_fashion_mnist_vae(; strong=true)
    if strong
        vae = deserialize(joinpath(vision_dir, "fashion_mnist_vae_strong.jls"))
    else
        vae = deserialize(joinpath(vision_dir, "fashion_mnist_vae_weak.jls"))
    end
    return vae
end
