# Report: Integrating MLJ models into the package


## Generator compatibility analysis for each MLJ model

The tables below will present an overview of the compatibility of models from the [MLJ general registry](https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/) with ``CounterfactualExplanations.jl``, organized by the interface library.


### BetaML.jl

| **Model** | **Analysis** |
| -------- | ------- |
| DecisionTreeClassifier, RandomForestClassifier | Currently incompatible with the gradient-based generators implemented in our package. However, counterfactual explanations can be generated for these models using the Feature Tweak generator we implemented. The models could also be made compatible with gradient-based generators in the future, possibly through the use of [probability calibration](https://scikit-learn.org/stable/modules/calibration.html). |
| NeuralNetworkClassifier, LinearPerceptron | Currently incompatible with all generators. However, it should be possible to make these models compatible with all gradient-based generators once the best way to access their gradients has been figured out: all gradient-based generators need gradient-access and currently leverage `Zygote.jl` for that, but `Zygote.jl` does not work for MLJ models, so an alternative solution has to be found. |
| KernelPerceptron | Currently incompatible with all generators, since the model is not differentiable. The possibility of using probability calibration should be explored. |

The library also contains various regressors and unsupervised models which are incompatible with all generators, as our package currently only supports generating counterfactual explanations for classifiers.


### CatBoost.jl

| **Model** | **Analysis** |
| -------- | ------- |
| CatBoostClassifier | Currently incompatible with all generators. However, the model relies on gradient-boosted decision trees in a way that is seemingly highly similar to EvoTrees, so the possibility of supporting the model should be explored after implementing support for EvoTrees. |
| CatBoostRegressor | Incompatible with all generators, as the package currently only supports generating counterfactual explanations for classification models. |


### MLJDecisionTreeInterface.jl

| **Model** | **Analysis** |
| -------- | ------- |
| DecisionTreeClassifier, RandomForestClassifier | Currently incompatible with gradient-based generators. However, counterfactual explanations can be generated for this model using the Feature Tweak generator we implemented. The models could also be made compatible with gradient-based generators in the future, possibly through the use of [probability calibration](https://scikit-learn.org/stable/modules/calibration.html). |
| DecisionTreeRegressor, RandomForestRegressor | Incompatible with all generators, as the package currently only supports generating counterfactual explanations for classifiers. |
| AdaBoostClassifier | Currently not supported, but the compatibility of the model with both gradient-based generators as well as with Feature Tweak is worth exploring. |


### EvoTrees.jl

| **Model** | **Analysis** |
| -------- | ------- |
| EvoTreeClassifier | We are currently in the process of implementing support for this model. |
| EvoTreeRegressor, EvoTreeCount, EvoTreeGaussian, EvoTreeMLE | Incompatible with all generators, as the package currently only supports generating counterfactual explanations for classifiers. |


### MLJLinearModels.jl

| **Model** | **Analysis** |
| -------- | ------- |
| LogisticClassifier, MultinomialClassifier | Both models are differentiable and it should be possible to make them compatible with the package. |

The library also contains various regressors which are incompatible with all generators, as our package currently only supports generating counterfactual explanations for classifiers.


### MLJGLMInterface.jl

| **Model** | **Analysis** |
| -------- | ------- |
| LinearBinaryClassifier | The model is differentiable, but given that linear classifiers are offered by other MLJ-supported libraries with higher maturity and that this is the only model from this library compatible with our package, it is unclear whether we should offer support for this model. |

The library also contains various regressors which are incompatible with all generators, as our package currently only supports generating counterfactual explanations for classifiers.


### MLJLIBSVMInterface.jl

| **Model** | **Analysis** |
| -------- | ------- |
| LinearSVC, SVC, NuSVC | Though Support Vector Classifiers (SVCs) are not differentiable in general, there have been recent efforts to propose counterfactual generators for such models: see, e.g., ["Counterfactual Explanations for Support Vector Machine Models"](https://arxiv.org/abs/2212.07432) by Salazar et al. However, adding support for these kinds of models requires implementing new counterfactual generators, which is infeasible for us to do in the scope of the software project, as we're already working on various other counterfactual generators. Furthermore, research in this area seems to be very recent, so it might be better to wait with implementing these generators until further research is published and generators such as the one proposed in the paper by Salazar et al. become more mature. Nevertheless, adding support for these models seems like a promising future direction for the package. |
| NuSVR, EpsilonSVR, OneClassSVM | Since these models are either regression or unsupervised models, support for them will not be implemented at the moment. |


### MLJMultivariateStatsInterface.jl

| **Model** | **Analysis** |
| -------- | ------- |
| LDA, BayesianLDA, SubspaceLDA, BayesianSubspaceLDA | These are all statistical models for which numerical optimization techniques are commonly not used. However, all of these models have linear decision boundaries, which means that it is possible to compute gradients of these models' decision functions with respect to their input. This makes them compatible with the package's existing counterfactual generators in theory. In practice, however, they are very different from all of the models for which the generators are used for at the moment, so some implementation difficulties should be expected. |

The library also contains various regressors and unsupervised models which are incompatible with all generators, as our package currently only supports generating counterfactual explanations for classifiers.


### NearestNeighborModels.jl

| **Model** | **Analysis** |
| -------- | ------- |
| KNNClassifier, MultitargetKNNClassifier | k-Nearest Neighbour classifiers do not have a differentiable decision boundary, so they are incompatible with the generators currently implemented in our package. We were unable to find papers that explicitly tackle the problem of making such models differentiable for the purpose of generating counterfactuals for them, so we expect the task of making them compatible with the package to be very challenging and outside of the scope of the software project. |
| KNNRegressor, MultitargetKNNRegressor | As these models are regression models, they are incompatible with the generators currently implemented in the package. |


### OneRule.jl, MLJText.jl, MLJNaiveBayesInterface.jl, PartialLeastSquaresRegressor.jl

As the maturity of each of these packages is marked as either low or experimental in the [MLJ model registry](https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/), the compatibility of models from these libraries with CounterfactualExplanations.jl will not be explored during the software project, as these libraries are expected to change a lot in the near future and this could require constant changes of the code in our library to maintain the compatibility with these models. Instead, it is better to wait until the libraries become more mature.


### MLJFlux.jl

Though the maturity of this library is also marked to be low, we think that it's nevertheless worth an attempt to make our package compatible with models from this library. This is because the models are compatible with Zygote.jl, which is the library currently used for automatic differentiation for the models already implemented in our package.

| **Model** | **Analysis** |
| -------- | ------- |
| NeuralNetworkClassifier, ImageClassifier | Currently incompatible with the package, but [the client has already explored the possibility of incorporating these models into the package](https://github.com/FluxML/MLJFlux.jl/issues/220). We are planning to build upon that progress. |
| NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor | Incompatible with all generators, as the package currently only supports generating counterfactual explanations for classifiers. |


### EvoLinear.jl, PartialLeastSquaresRegressor.jl

All models from these packages are incompatible with all generators, since the packages only offer regression models, which the generators don't currently support.


### MLJClusteringInterface.jl, OutlierDetectionNeighbors.jl, OutlierDetectionNetworks.jl, OutlierDetectionPython.jl, ParallelKMeans.jl, TSVD.jl

All models from these packages are incompatible with all generators, since the packages only offer unsupervised models for which counterfactual explanations don't apply.


### ScikitLearn.jl, LightGBM.jl

Since the models offered by these libraries are not native to Julia (both are interfaces to Python models) and the task of generating counterfactuals for them is thus expected to be more difficult compared to native Julia models, we will explore the possible compatibility of models from this library once we have finished working on MLJ models natively implemented in Julia.


### MLJModels.jl

This is the base MLJ package from which all the other MLJ-supported models can be loaded. Support for the models that can be loaded through this library has been documented in the whole document above. The package also offers some models on its own, but the generation of counterfactuals for these models is not supported, as they are mostly helper models such as OneHotEncoder and UnivariateBoxCoxTransformer that don't make class predictions on their own.