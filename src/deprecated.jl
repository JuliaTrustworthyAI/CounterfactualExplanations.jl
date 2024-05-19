Base.@deprecate_moved embed_path "TaijaPlotting" false

Base.@deprecate_moved animate_path "TaijaPlotting" false

Base.@deprecate_moved voronoi "TaijaPlotting" false

Base.@deprecate_moved prepare_for_plotting "TaijaPlotting" false

Base.@deprecate_moved data_catalogue "TaijaData" false

Base.@deprecate_moved load_linearly_separable "TaijaData" false

Base.@deprecate_moved load_overlapping "TaijaData" false

Base.@deprecate_moved load_multi_class "TaijaData" false

Base.@deprecate_moved load_blobs "TaijaData" false

Base.@deprecate_moved load_circles "TaijaData" false

Base.@deprecate_moved load_moons "TaijaData" false

Base.@deprecate_moved load_synthetic_data "TaijaData" false

Base.@deprecate_moved load_california_housing "TaijaData" false

Base.@deprecate_moved load_credit_default "TaijaData" false

Base.@deprecate_moved load_gmsc "TaijaData" false

Base.@deprecate_moved load_german_credit "TaijaData" false

Base.@deprecate_moved load_uci_adult "TaijaData" false

Base.@deprecate_moved load_tabular_data "TaijaData" false

Base.@deprecate_moved load_mnist "TaijaData" false

Base.@deprecate_moved load_mnist_test "TaijaData" false

Base.@deprecate_moved load_fashion_mnist "TaijaData" false

Base.@deprecate_moved load_fashion_mnist_test "TaijaData" false

Base.@deprecate_moved load_cifar_10 "TaijaData" false

Base.@deprecate_moved load_cifar_10_test "TaijaData" false

Base.@deprecate_moved PyTorchModel "TaijaInteroperability" false

Base.@deprecate_moved pytorch_model_loader "TaijaInteroperability" false

Base.@deprecate_moved preprocess_python_data "TaijaInteroperability" false

Base.@deprecate_moved RTorchModel "TaijaInteroperability" false

Base.@deprecate_moved rtorch_model_loader "TaijaInteroperability" false

Base.@deprecate_moved parallelize "TaijaParallel" false

Base.@deprecate_moved parallelizable "TaijaParallel" false

Base.@deprecate_moved AbstractParallelizer "TaijaBase" false

Base.@deprecate_binding AbstractFittedModel AbstractModel

Base.@deprecate FluxModel(model) MLP(model)

Base.@deprecate FluxModel(data::CounterfactualData) MLP()(data::CounterfactualData)

Base.@deprecate FluxEnsemble(model) DeepEnsemble(model)

Base.@deprecate FluxEnsemble(data::CounterfactualData) DeepEnsemble()(
    data::CounterfactualData
)

Base.@deprecate train!(vae, X, y) GenerativeModels.train!(vae, X)
