

``` @meta
CurrentModule = CounterfactualExplanations 
```

# Model Catalogue

While in general it is assumed that users will use this package to explain their pre-trained models, we provide out-of-the-box functionality to train various simple default models. In this tutorial, we will see how these models can be fitted to `CounterfactualData`.

## Available Models

The `standard_models_catalogue` can be used to inspect the available default models:

``` julia
standard_models_catalogue
```

    Dict{Symbol, Any} with 4 entries:
      :Linear       => Linear
      :LaplaceRedux => LaplaceReduxModel
      :DeepEnsemble => FluxEnsemble
      :MLP          => FluxModel

The dictionary keys correspond to the model names. In this case, the dictionary values are constructors that can be used called on instances of type `CounterfactualData` to fit the corresponding model. In most cases, users will find it most convenient to use the [`fit_model`](@ref) API call instead.

## Fitting Models

Models from the standard model catalogue are a core part of the package and thus compatible with all offered counterfactual generators and functionalities.

The `all_models_catalogue` can be used to inspect all models offered by the package:

``` julia
all_models_catalogue
```

However, when using models not included in the `standard_models_catalogue`, additional caution is advised: they might not be supported by all counterfactual generators or they might not be models native to Julia. Thus, a more thorough reading of their documentation may be necessary to make sure that they are used correctly.

## Fitting Flux Models

First, let’s load one of the synthetic datasets. For this, we’ll first need to import the `TaijaData.jl` package:

``` julia
n = 500
data = TaijaData.load_multi_class(n)
counterfactual_data = DataPreprocessing.CounterfactualData(data...)
```

We could use a Deep Ensemble (Lakshminarayanan, Pritzel, and Blundell 2016) as follows:

``` julia
M = fit_model(counterfactual_data, :DeepEnsemble)
```

The returned object is an instance of type `FluxEnsemble <: AbstractFittedModel` and can be used in downstream tasks without further ado. For example, the resulting fit can be visualised using the generic `plot()` method as:

``` julia
plts = []
for target in counterfactual_data.y_levels
    plt = plot(M, counterfactual_data; target=target, title="p(y=$(target)|x,θ)")
    plts = [plts..., plt]
end
plot(plts...)
```

![](model_catalogue_files/figure-commonmark/cell-7-output-1.svg)

## Importing PyTorch models

The package supports generating counterfactuals for any neural network that has been previously defined and trained using PyTorch, regardless of the specific architectural details of the model. To generate counterfactuals for a PyTorch model, save the model inside a `.pt` file and call the following function:

``` julia
model_loaded = TaijaInteroperability.pytorch_model_loader(
    "$(pwd())/docs/src/tutorials/miscellaneous",
    "neural_network_class",
    "NeuralNetwork",
    "$(pwd())/docs/src/tutorials/miscellaneous/pretrained_model.pt"
)
```

The method `pytorch_model_loader` requires four arguments:
1. The path to the folder with a `.py` file where the PyTorch model is defined
2. The name of the file where the PyTorch model is defined
3. The name of the class of the PyTorch model
4. The path to the Pickle file that holds the model weights

In the above case:
1. The file defining the model is inside `$(pwd())/docs/src/tutorials/miscellaneous`
2. The name of the `.py` file holding the model definition is `neural_network_class`
3. The name of the model class is NeuralNetwork
4. The Pickle file is located at `$(pwd())/docs/src/tutorials/miscellaneous/pretrained_model.pt`

Though the model file and Pickle file are inside the same directory in this tutorial, this does not necessarily have to be the case.

The reason why the model file and Pickle file have to be provided separately is that the package expects an already trained PyTorch model as input. It is also possible to define new PyTorch models within the package, but since this is not the expected use of our package, special support is not offered for that. A guide for defining Python and PyTorch classes in Julia through `PythonCall.jl` can be found [here](https://cjdoris.github.io/PythonCall.jl/stable/pythoncall-reference/#Create-classes).

Once the PyTorch model has been loaded into the package, wrap it inside the PyTorchModel class:

``` julia
model_pytorch = TaijaInteroperability.PyTorchModel(model_loaded, counterfactual_data.likelihood)
```

This model can now be passed into the generators like any other.

Please note that the functionality for generating counterfactuals for Python models is only available if your Julia version is 1.8 or above. For Julia 1.7 users, we recommend upgrading the version to 1.8 or 1.9 before loading a PyTorch model into the package.

## Importing R torch models

!!! warning "Not fully tested"
    Please note that due to the incompatibility between RCall and PythonCall, it is not feasible to test both PyTorch and RTorch implementations within the same pipeline. While the RTorch implementation has been manually tested, we cannot ensure its consistent functionality as it is inherently susceptible to bugs.

The CounterfactualExplanations package supports generating counterfactuals for neural networks that have been defined and trained using R torch. Regardless of the specific architectural details of the model, you can easily generate counterfactual explanations by following these steps.

### Saving the R torch model

First, save your trained R torch model as a `.pt` file using the `torch_save()` function provided by the R torch library. This function allows you to serialize the model and save it to a file. For example:

``` r
torch_save(model, file = "$(pwd())/docs/src/tutorials/miscellaneous/r_model.pt")
```

Make sure to specify the correct file path where you want to save the model.

### Loading the R torch model

To import the R torch model into the CounterfactualExplanations package, use the `rtorch_model_loader()` function. This function loads the model from the previously saved `.pt` file. Here is an example of how to load the R torch model:

``` julia
model_loaded = TaijaInteroperability.rtorch_model_loader("$(pwd())/docs/src/tutorials/miscellaneous/r_model.pt")
```

The `rtorch_model_loader()` function requires only one argument:
1. `model_path`: The path to the `.pt` file that contains the trained R torch model.

### Wrapping the R torch model

Once the R torch model has been loaded into the package, wrap it inside the `RTorchModel` class. This step prepares the model to be used by the counterfactual generators. Here is an example:

``` julia
model_R = TaijaInteroperability.RTorchModel(model_loaded, counterfactual_data.likelihood)
```

### Generating counterfactuals with the R torch model

Now that the R torch model has been wrapped inside the `RTorchModel` class, you can pass it into the counterfactual generators as you would with any other model.

Please note that RCall is not fully compatible with PythonCall. Therefore, it is advisable not to import both R torch and PyTorch models within the same Julia session. Additionally, it’s worth mentioning that the R torch integration is still untested in the CounterfactualExplanations package.

## Tuning Flux Models

By default, model architectures are very simple. Through optional arguments, users have some control over the neural network architecture and can choose to impose regularization through dropout. Let’s tackle a more challenging dataset: MNIST (LeCun 1998).

``` julia
data = TaijaData.load_mnist(10000)
counterfactual_data = DataPreprocessing.CounterfactualData(data...)
train_data, test_data = 
    CounterfactualExplanations.DataPreprocessing.train_test_split(counterfactual_data)
```

![](model_catalogue_files/figure-commonmark/cell-9-output-1.svg)

In this case, we will use a Multi-Layer Perceptron (MLP) but we will adjust the model and training hyperparameters. Parameters related to training of `Flux.jl` models are currently stored in a mutable container:

``` julia
flux_training_params
```

    CounterfactualExplanations.FluxModelParams
      loss: Symbol logitbinarycrossentropy
      opt: Symbol Adam
      n_epochs: Int64 100
      batchsize: Int64 1
      verbose: Bool false

In cases like this one, where model training can be expected to take a few moments, it can be useful to activate verbosity, so let’s set the corresponding field value to `true`. We’ll also impose mini-batch training:

``` julia
flux_training_params.verbose = true
flux_training_params.batchsize = round(size(train_data.X,2)/10)
```

To account for the fact that this is a slightly more challenging task, we will use an appropriate number of hidden neurons per layer. We will also activate dropout regularization. To scale networks up further, it is also possible to adjust the number of hidden layers, which we will not do here.

``` julia
model_params = (
    n_hidden = 32,
    dropout = true
)
```

The `model_params` can be supplied to the familiar API call:

``` julia
M = fit_model(train_data, :MLP; model_params...)
```

    FluxModel(Chain(Dense(784 => 32, relu), Dropout(0.25, active=false), Dense(32 => 10)), :classification_multi)

The model performance on our test set can be evaluated as follows:

``` julia
model_evaluation(M, test_data)
```

    1-element Vector{Float64}:
     0.9185

Finally, let’s restore the default training parameters:

``` julia
CounterfactualExplanations.reset!(flux_training_params)
```

## Fitting and tuning MLJ models

Among models from the MLJ library, two models are integrated as part of the core functionality of the package:

``` julia
mlj_models_catalogue
```

These models are compatible with the Feature Tweak generator. Support for other generators has not been implemented, as both decision trees and random forests are non-differentiable tree-based models and thus, gradient-based generators don’t apply for them.

Tuning MLJ models is very simple. As the first step, let’s reload the dataset:

``` julia
n = 500
data = TaijaData.load_moons(n)
counterfactual_data = DataPreprocessing.CounterfactualData(data...)
```

Using the usual procedure for fitting models, we can call the following method:

``` julia
tree = CounterfactualExplanations.Models.fit_model(counterfactual_data, :DecisionTree)
```

However, it’s also possible to tune the DecisionTreeClassifier’s parameters. This can be done using the keyword arguments when calling `fit_model()` as follows:

``` julia
tree = CounterfactualExplanations.Models.fit_model(counterfactual_data, :DecisionTree; max_depth=2, min_samples_leaf=3)
```

For all supported MLJ models, every tunable parameter they have is supported as a keyword argument. The tunable parameters for the `DecisionTreeModel` and the `RandomForestModel` can be found from the [documentation of the `DecisionTree.jl` package](https://docs.juliahub.com/DecisionTree/pEDeB/0.10.11/) under the Decision Tree Classifier and Random Forest Classifier sections.

## Package extension models

The package also includes two models which don’t form a part of the core functionality of the package, but which can be accessed as package extensions. These are the `EvoTreeModel` from the MLJ library and the `LaplaceReduxModel` from `LaplaceRedux.jl`.

To trigger the package extensions, the weak dependency first has to be loaded with the `using` keyword:

``` julia
using EvoTrees
```

Once this is done, the extension models can be used like any other model:

``` julia
M = fit_model(counterfactual_data, :EvoTree; model_params...)
```

    EvoTreesExt.EvoTreeModel(machine(EvoTreeClassifier{EvoTrees.MLogLoss}
     - nrounds: 100
     - L2: 0.0
     - lambda: 0.0
     - gamma: 0.0
     - eta: 0.1
     - max_depth: 6
     - min_weight: 1.0
     - rowsample: 1.0
     - colsample: 1.0
     - nbins: 64
     - alpha: 0.5
     - tree_type: binary
     - rng: MersenneTwister(123, (0, 9018, 8016, 884))
    , …), :classification_multi)

The tunable parameters for the `EvoTreeModel` can be found from the [documentation of the `EvoTrees.jl` package](https://evovest.github.io/EvoTrees.jl/stable/) under the EvoTreeClassifier section.

Please note that support for counterfactual generation with both `LaplaceReduxModel` and `EvoTreeModel` is not yet fully implemented.

## References

Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. 2016. “Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles.” <https://arxiv.org/abs/1612.01474>.

LeCun, Yann. 1998. “The MNIST Database of Handwritten Digits.”
