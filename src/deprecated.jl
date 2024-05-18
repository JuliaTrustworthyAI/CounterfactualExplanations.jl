Base.@deprecate_moved embed_path "TaijaPlotting"

Base.@deprecate_moved animate_path "TaijaPlotting"

Base.@deprecate_moved voronoi "TaijaPlotting"

Base.@deprecate_moved prepare_for_plotting "TaijaPlotting"

Base.@deprecate_moved data_catalogue "TaijaData"

Base.@deprecate_moved load_linearly_separable "TaijaData"

Base.@deprecate_moved load_overlapping "TaijaData"

Base.@deprecate_moved load_multi_class "TaijaData"

Base.@deprecate_moved load_blobs "TaijaData"

Base.@deprecate_moved load_circles "TaijaData"

Base.@deprecate_moved load_moons "TaijaData"

Base.@deprecate_moved load_synthetic_data "TaijaData"

Base.@deprecate_moved load_california_housing "TaijaData"

Base.@deprecate_moved load_credit_default "TaijaData"

Base.@deprecate_moved load_gmsc "TaijaData"

Base.@deprecate_moved load_german_credit "TaijaData"

Base.@deprecate_moved load_uci_adult "TaijaData"

Base.@deprecate_moved load_tabular_data "TaijaData"

Base.@deprecate_moved load_mnist "TaijaData"

Base.@deprecate_moved load_mnist_test "TaijaData"

Base.@deprecate_moved load_fashion_mnist "TaijaData"

Base.@deprecate_moved load_fashion_mnist_test "TaijaData"

Base.@deprecate_moved load_cifar_10 "TaijaData"

Base.@deprecate_moved load_cifar_10_test "TaijaData"

Base.@deprecate_moved PyTorchModel "TaijaInteroperability"

Base.@deprecate_moved pytorch_model_loader "TaijaInteroperability"

Base.@deprecate_moved preprocess_python_data "TaijaInteroperability"

Base.@deprecate_moved RTorchModel "TaijaInteroperability"

Base.@deprecate_moved rtorch_model_loader "TaijaInteroperability"

Base.@deprecate_moved parallelize "TaijaParallel"

Base.@deprecate_moved parallelizable "TaijaParallel"

Base.@deprecate_moved AbstractParallelizer "TaijaBase"

Base.@deprecate_binding AbstractFittedModel AbstractModel

Base.@deprecate FluxModel(model) MLP(model)

Base.@deprecate FluxModel(data::CounterfactualData) MLP()(data::CounterfactualData)

Base.@deprecate FluxEnsemble(model) DeepEnsemble(model)

Base.@deprecate FluxEnsemble(data::CounterfactualData) DeepEnsemble()(
    data::CounterfactualData
)

Base.@deprecate train!(vae, X, y) GenerativeModels.train!(vae, X)
