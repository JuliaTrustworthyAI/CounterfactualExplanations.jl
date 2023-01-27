using LazyArtifacts
using Serialization

vision_dir = joinpath(artifact"model-vision", "model-vision")
const artifacts_warning = "Pre-trained models have been serialized and may not be compatible depending on which Julia version you are using. We originally relied on BSON, but this too led to issues."

function load_mnist_mlp()
    @warn artifacts_warning
    M = deserialize(joinpath(vision_dir,"mnist_mlp.jls"))
    return M
end

function load_mnist_ensemble()
    @warn artifacts_warning
    M = deserialize(joinpath(vision_dir, "mnist_ensemble.jls"))
    return M
end

function load_mnist_vae()
    @warn artifacts_warning
    vae = deserialize(joinpath(vision_dir, "mnist_vae.jls"))
    return vae
end