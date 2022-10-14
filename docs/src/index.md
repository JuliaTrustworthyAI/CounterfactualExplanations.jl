
``` @meta
CurrentModule = CounterfactualExplanations
```

# CounterfactualExplanations

Documentation for [CounterfactualExplanations.jl](https://github.com/pat-alt/CounterfactualExplanations.jl).

`CounterfactualExplanations.jl` is a package for generating Counterfactual Explanations (CE) and Algorithmic Recourse (AR) for black-box algorithms. Both CE and AR are related tools for explainable artificial intelligence (XAI). While the package is written purely in Julia, it can be used to explain machine learning algorithms developed and trained in other popular programming languages like Python and R. See below for short introduction and other resources or dive straight into the [docs](https://pat-alt.github.io/CounterfactualExplanations.jl/dev).

## News 📣

``` julia
versioninfo()
```

    Julia Version 1.8.1
    Commit afb6c60d69a (2022-09-06 15:09 UTC)
    Platform Info:
      OS: macOS (x86_64-apple-darwin21.4.0)
      CPU: 16 × Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
      WORD_SIZE: 64
      LIBM: libopenlibm
      LLVM: libLLVM-13.0.1 (ORCJIT, skylake)
      Threads: 1 on 8 virtual cores
    Environment:
      JULIA_EDITOR = code

**JuliaCon 2022**: This project was presented at JuliaCon 2022 in July 2022. See [here](https://pretalx.com/juliacon-2022/talk/HU8FVH/) for details.

### Developmemts since v0.1.2

- Added support for generating multiple counterfactuals instead of just one.
- Implemented DiCE (Mothilal, Sharma, and Tan 2020).
- Added option to add a random perturbation to the initial factual values, which is an easy way to mitigate adversarial attacks on CE (Slack et al. 2021).
- Visualizations: added various plotting methods to visualize results. For *n*-dimensional data where *n* \> 2 we use dimensionality reduction to embed the data in 2D. The decision boundary is then visualized using its a Voronoi based representation following Migut, Worring, and Veenman (2015). Thanks to my student [Aleksander Buszydlik](https://github.com/abuszydlik) for bringing this idea to my attention.
- Added the option to specify strict convergence: instead of terminating search once the probability threshold *γ* has been reached, search continues as long the objective function still improves.

## Installation 🚩

The first release of this package is now on [Julia’s General Registry](https://github.com/JuliaRegistries/General) and can be installed as follows:

``` julia
using Pkg
Pkg.add("CounterfactualExplanations")
```

CounterfactualExplanations.jl is currently under active development. To install the development version of the package you can run the following command:

``` julia
using Pkg
Pkg.add(url="https://github.com/pat-alt/CounterfactualExplanations.jl")
```

## Background and motivation

#### The Need for Explainability ⬛

Machine learning models like deep neural networks have become so complex, opaque and underspecified in the data that they are generally considered as black boxes. Nonetheless, such models often play a key role in data-driven decision-making systems. This often creates the following problem: human operators in charge of such systems have to rely on them blindly, while those individuals subject to them generally have no way of challenging an undesirable outcome:

> “You cannot appeal to (algorithms). They do not listen. Nor do they bend.”
>
> — Cathy O’Neil in [*Weapons of Math Destruction*](https://en.wikipedia.org/wiki/Weapons_of_Math_Destruction), 2016

#### Enter: Counterfactual Explanations 🔮

Counterfactual Explanations can help human stakeholders make sense of the systems they develop, use or endure: they explain how inputs into a system need to change for it to produce different decisions. Explainability benefits internal as well as external quality assurance. The figure below, for example, shows various counterfactuals generated through different approaches that all turn the predicted label of some classifier from a 9 into a 4. CEs that involve realistic and actionable changes such as the one on the far right can be used for the purpose of algorithmic recourse.

![Realistic counterfactual explanations for MNIST data: turning a 4 into a 9.](https://raw.githubusercontent.com/pat-alt/CounterfactualExplanations.jl/main/docs/src/examples/image/www/MNIST_9to4.png)

Explanations that involve realistic and actionable changes can be used for the purpose of algorithmic recourse (AR): they offer human stakeholders a way to not only understand the system’s behaviour, but also react to it or adjust it. The figure below illustrates the point of AR through a toy example: it shows the counterfactual path of one sad cat 🐱 that would like to be grouped with her cool dog friends. Unfortunately, based on her tail length and height she was classified as a cat by a black-box classifier. The recourse algorithm perturbs her features in such a way that she ends up crossing the decision boundary into a dense region inside the target class.

![A sad 🐱 on its counterfactual path to its cool dog friends.](https://raw.githubusercontent.com/pat-alt/CounterfactualExplanations.jl/main/docs/src/www/recourse_laplace.gif)

Counterfactual Explanations have certain advantages over related tools for explainable artificial intelligence (XAI) like surrogate eplainers (LIME and SHAP). These include:

- Full fidelity to the black-box model, since no proxy is involved.
- No need for (reasonably) interpretable features as opposed to LIME and SHAP.
- Clear link to Causal Inference and Bayesian Machine Learning.
- Less susceptible to adversarial attacks than LIME and SHAP.

## Usage example 🔍

Generating counterfactuals will typically look like follows:

``` julia
using CounterfactualExplanations

# Data:
using CounterfactualExplanations.Data
using Random
Random.seed!(1234)
xs, ys = Data.toy_data_linear()
X = hcat(xs...)
counterfactual_data = CounterfactualData(X,ys')

# Model
using CounterfactualExplanations.Models: LogisticModel, probs 
# Logit model:
w = [1.0 1.0] # true coefficients
b = 0
M = LogisticModel(w, [b])

# Randomly selected factual:
x = select_factual(counterfactual_data,rand(1:size(X)[2]))
y = round(probs(M, x)[1])
target = round(probs(M, x)[1])==0 ? 1 : 0 

# Counterfactual search:
generator = GenericGenerator()
```

Running the counterfactual search yields:

``` julia
counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator)
```

    Convergence: ✅

     after 29 steps.

## Implemented Counterfactual Generators:

Currently the following counterfactual generators are implemented:

- Generic (Wachter, Mittelstadt, and Russell 2017)
- Greedy (Schut et al. 2021)
- DiCE (Mothilal, Sharma, and Tan 2020)
- Latent Space Search as in REVISE (Joshi et al. 2019) and CLUE (Antorán et al. 2020)

## Goals and limitations 🎯

The goal for this library is to contribute to efforts towards trustworthy machine learning in Julia. The Julia language has an edge when it comes to trustworthiness: it is very transparent. Packages like this one are generally written in pure Julia, which makes it easy for users and developers to understand and contribute to open source code. Eventually the aim for this project is to offer a one-stop-shop of counterfactual explanations. We want to deliver a package that is at least at par with the [CARLA](https://github.com/carla-recourse/CARLA) Python library in terms of its functionality. Contrary to CARLA, we aim for languague interoperability in the long-term. Currently the package falls short of this goal in a number of ways: 1) the number of counterfactual generators is limited, 2) the data preprocessing functionality needs to be extended.

## Contribute 🛠

Contributions of any kind are very much welcome! If any of the below applies to you, this might be the right open-source project for you:

- You’re an expert in Counterfactual Explanations or Explainable AI more broadly and you are curious about Julia.
- You’re an experienced Julian and are happy to help a less experienced Julian (yours truly) up their game. Ideally, you are also curious about Trustworthy AI.
- You’re new to Julia and open-source development and would like to start your learning journey by contributing to a recent but promising development. Ideally you are familiar with machine learning.

I am still very much at the beginning of my Julia journey, so if you spot any issues or have any suggestions for design improvement, please just open [issue](https://github.com/pat-alt/CounterfactualExplanations.jl/issues) or start a [discussion](https://github.com/pat-alt/CounterfactualExplanations.jl/discussions). Our goal is to provide a go-to place for counterfactual explanations in Julia. To this end, the following is a non-exhaustive list of exciting feature developments we envision:

1.  Additional counterfactual generators and predictive models.
2.  Additional datasets for testing, evaluation and benchmarking.
3.  Improved preprocessing including native support for categorical features.
4.  Support for regression models.

For more details on how to contribute see [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/contributing/). Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## Citation 🎓

If you want to use this codebase, please consider citing:

    @software{altmeyer2022CounterfactualExplanations,
      author = {Patrick Altmeyer},
      title = {{CounterfactualExplanations.jl - a Julia package for Counterfactual Explanations and Algorithmic Recourse}},
      url = {https://github.com/pat-alt/CounterfactualExplanations.jl},
      version = {0.1.1},
      year = {2022}
    }

## References 📚

Antorán, Javier, Umang Bhatt, Tameem Adel, Adrian Weller, and José Miguel Hernández-Lobato. 2020. “Getting a Clue: A Method for Explaining Uncertainty Estimates.” <https://arxiv.org/abs/2006.06848>.

Joshi, Shalmali, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh. 2019. “Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems.” <https://arxiv.org/abs/1907.09615>.

Migut, MA, Marcel Worring, and Cor J Veenman. 2015. “Visualizing Multi-Dimensional Decision Boundaries in 2D.” *Data Mining and Knowledge Discovery* 29 (1): 273–95.

Mothilal, Ramaravind K, Amit Sharma, and Chenhao Tan. 2020. “Explaining Machine Learning Classifiers Through Diverse Counterfactual Explanations.” In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607–17.

Schut, Lisa, Oscar Key, Rory Mc Grath, Luca Costabello, Bogdan Sacaleanu, Yarin Gal, et al. 2021. “Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties.” In *International Conference on Artificial Intelligence and Statistics*, 1756–64. PMLR.

Slack, Dylan, Anna Hilgard, Himabindu Lakkaraju, and Sameer Singh. 2021. “Counterfactual Explanations Can Be Manipulated.” *Advances in Neural Information Processing Systems* 34.

Wachter, Sandra, Brent Mittelstadt, and Chris Russell. 2017. “Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR.” *Harv. JL & Tech.* 31: 841.
