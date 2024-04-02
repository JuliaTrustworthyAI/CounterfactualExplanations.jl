vision_dir = CounterfactualExplanations.generate_artifact_dir("model-vision")

include("cifar_10.jl")
include("fashion_mnist.jl")
include("fomc.jl")
include("mnist.jl")
