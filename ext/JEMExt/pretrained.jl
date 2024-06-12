using Serialization

vision_dir = CounterfactualExplanations.generate_artifact_dir("model-vision")

"""
    Models.load_mnist_model(type::CounterfactualExplanations.JEM)

Overload for loading a pre-trained model for the `JEM` model type.
"""
function Models.load_mnist_model(type::CounterfactualExplanations.JEM)
    model = Serialization.deserialize(joinpath(vision_dir, "mnist_jem.jls"))

    M = type(model[1]; likelihood=:classification_multi)
    return M
end
