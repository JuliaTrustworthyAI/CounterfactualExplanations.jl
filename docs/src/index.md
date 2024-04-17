

``` @meta
CurrentModule = CounterfactualExplanations
```

![](assets/wide_logo.png)

Documentation for [CounterfactualExplanations.jl](https://github.com/juliatrustworthyai/CounterfactualExplanations.jl).

# CounterfactualExplanations

*Counterfactual Explanations and Algorithmic Recourse in Julia.*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/dev) [![Build Status](https://github.com/juliatrustworthyai/CounterfactualExplanations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/juliatrustworthyai/CounterfactualExplanations.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/juliatrustworthyai/CounterfactualExplanations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/juliatrustworthyai/CounterfactualExplanations.jl) [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![License](https://img.shields.io/github/license/juliatrustworthyai/CounterfactualExplanations.jl)](assets/intro.gif) [![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/CounterfactualExplanations/.png)](https://pkgs.genieframework.com?packages=CounterfactualExplanations) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

`CounterfactualExplanations.jl` is a package for generating Counterfactual Explanations (CE) and Algorithmic Recourse (AR) for black-box algorithms. Both CE and AR are related tools for explainable artificial intelligence (XAI). While the package is written purely in Julia, it can be used to explain machine learning algorithms developed and trained in other popular programming languages like Python and R. See below for a short introduction and other resources or dive straight into the [docs](https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/dev).

There is also a corresponding paper, [*Explaining Black-Box Models through Counterfactuals*](https://proceedings.juliacon.org/papers/10.21105/jcon.00130), which has been published in JuliaCon Proceedings. Please consider citing the paper, if you use this package in your work:

[![DOI](https://proceedings.juliacon.org/papers/10.21105/jcon.00130/status.svg)](https://doi.org/10.21105/jcon.00130) [![DOI](https://zenodo.org/badge/440782065.svg)](https://zenodo.org/badge/latestdoi/440782065)

    @article{Altmeyer2023,
      doi = {10.21105/jcon.00130},
      url = {https://doi.org/10.21105/jcon.00130},
      year = {2023},
      publisher = {The Open Journal},
      volume = {1},
      number = {1},
      pages = {130},
      author = {Patrick Altmeyer and Arie van Deursen and Cynthia C. s. Liem},
      title = {Explaining Black-Box Models through Counterfactuals},
      journal = {Proceedings of the JuliaCon Conferences}
    }

## 🚩 Installation

You can install the stable release from [Julia’s General Registry](https://github.com/JuliaRegistries/General) as follows:

``` julia
using Pkg
Pkg.add("CounterfactualExplanations")
```

`CounterfactualExplanations.jl` is under active development. To install the development version of the package you can run the following command:

``` julia
using Pkg
Pkg.add(url="https://github.com/juliatrustworthyai/CounterfactualExplanations.jl")
```

## 🤔 Background and Motivation

Machine learning models like Deep Neural Networks have become so complex, opaque and underspecified in the data that they are generally considered Black Boxes. Nonetheless, such models often play a key role in data-driven decision-making systems. This creates the following problem: human operators in charge of such systems have to rely on them blindly, while those individuals subject to them generally have no way of challenging an undesirable outcome:

> “You cannot appeal to (algorithms). They do not listen. Nor do they bend.”
>
> — Cathy O’Neil in [*Weapons of Math Destruction*](https://en.wikipedia.org/wiki/Weapons_of_Math_Destruction), 2016

## 🔮 Enter: Counterfactual Explanations

Counterfactual Explanations can help human stakeholders make sense of the systems they develop, use or endure: they explain how inputs into a system need to change for it to produce different decisions. Explainability benefits internal as well as external quality assurance.

Counterfactual Explanations have a few properties that are desirable in the context of Explainable Artificial Intelligence (XAI). These include:

- Full fidelity to the black-box model, since no proxy is involved.
- No need for (reasonably) interpretable features as opposed to LIME and SHAP.
- Clear link to Algorithmic Recourse and Causal Inference.
- Less susceptible to adversarial attacks than LIME and SHAP.

### Example: Give Me Some Credit

Consider the following real-world scenario: a retail bank is using a black-box model trained on their clients’ credit history to decide whether they will provide credit to new applicants. To simulate this scenario, we have pre-trained a binary classifier on the publicly available Give Me Some Credit dataset that ships with this package (Kaggle 2011).

The figure below shows counterfactuals for 10 randomly chosen individuals that would have been denied credit initially.

![](index_files/figure-commonmark/cell-5-output-1.svg)

### Example: MNIST

The figure below shows a counterfactual generated for an image classifier trained on MNIST: in particular, it demonstrates which pixels need to change in order for the classifier to predict 3 instead of 8.

Since `v0.1.9` counterfactual generators are fully composable. Here we have composed a generator that combines ideas from Wachter, Mittelstadt, and Russell (2017) and Altmeyer et al. (2023):

``` julia
# Compose generator:
using CounterfactualExplanations.Objectives: distance_from_target
generator = GradientBasedGenerator()
@chain generator begin
    @objective logitcrossentropy + 0.1distance_mad + 0.1distance_from_target
    @with_optimiser Adam(0.1)                  
end
counterfactual_data.generative_model = vae # assign generative model
```

![](index_files/figure-commonmark/cell-10-output-1.svg)

## 🔍 Usage example

Generating counterfactuals will typically look like follows. Below we first fit a simple model to a synthetic dataset with linearly separable features and then draw a random sample:

``` julia
# Data and Classifier:
counterfactual_data = CounterfactualData(load_linearly_separable()...)
M = fit_model(counterfactual_data, :Linear)

# Select random sample:
target = 2
factual = 1
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)
```

To this end, we specify a counterfactual generator of our choice:

``` julia
# Counterfactual search:
generator = DiCEGenerator(λ=[0.1,0.3])
```

Here, we have chosen to use the `GradientBasedGenerator` to move the individual from its factual label 1 to the target label 2.

With all of our ingredients specified, we finally generate counterfactuals using a simple API call:

``` julia
conv = conv = CounterfactualExplanations.Convergence.GeneratorConditionsConvergence()
ce = generate_counterfactual(
  x, target, counterfactual_data, M, generator; 
  num_counterfactuals=3, convergence=conv,
)
```

The plot below shows the resulting counterfactual path:

![](index_files/figure-commonmark/cell-15-output-1.svg)

## ☑️ Implemented Counterfactual Generators

Currently, the following counterfactual generators are implemented:

- ClaPROAR (Altmeyer et al. 2023)
- CLUE (Antorán et al. 2020)
- DiCE (Mothilal, Sharma, and Tan 2020)
- FeatureTweak (Tolomei et al. 2017)
- Generic
- GravitationalGenerator (Altmeyer et al. 2023)
- Greedy (Schut et al. 2021)
- GrowingSpheres (Laugel et al. 2017)
- PROBE (Pawelczyk et al. 2023)
- REVISE (Joshi et al. 2019)
- Wachter (Wachter, Mittelstadt, and Russell 2017)

## 🎯 Goals and limitations

The goal of this library is to contribute to efforts towards trustworthy machine learning in Julia. The Julia language has an edge when it comes to trustworthiness: it is very transparent. Packages like this one are generally written in pure Julia, which makes it easy for users and developers to understand and contribute to open-source code. Eventually, this project aims to offer a one-stop-shop of counterfactual explanations.

Our ambition is to enhance the package through the following features:

1.  Support for all supervised machine learning models trained in [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/).
2.  Support for regression models.

## 🛠 Contribute

Contributions of any kind are very much welcome! Take a look at the [issue](https://github.com/juliatrustworthyai/CounterfactualExplanations.jl/issues) to see what things we are currently working on. If you have an idea for a new feature or want to report a bug, please open a new issue.

### Development

If your looking to contribute code, it may be helpful to check out the [Explanation](explanation/index.qmd) section of the docs.

#### Testing

Please always make sure to add tests for any new features or changes.

#### Documentation

If you add new features or change existing ones, please make sure to update the documentation accordingly. The documentation is written in [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/) and is located in the `docs/src` folder.

#### Log Changes

As of version `1.1.1`, we have tried to be more stringent about logging changes. Please make sure to add a note to the [CHANGELOG.md](CHANGELOG.md) file for any changes you make. It is sufficient to add a note under the `Unreleased` section.

### General Pointers

There are also some general pointers for people looking to contribute to any of our Taija packages [here](https://github.com/JuliaTrustworthyAI#general-pointers-for-contributors).

Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## 🎓 Citation

If you want to use this codebase, please consider citing the corresponding paper:

    @article{Altmeyer2023,
      doi = {10.21105/jcon.00130},
      url = {https://doi.org/10.21105/jcon.00130},
      year = {2023},
      publisher = {The Open Journal},
      volume = {1},
      number = {1},
      pages = {130},
      author = {Patrick Altmeyer and Arie van Deursen and Cynthia C. s. Liem},
      title = {Explaining Black-Box Models through Counterfactuals},
      journal = {Proceedings of the JuliaCon Conferences}
    }

## 📚 References

Altmeyer, Patrick, Giovan Angela, Aleksander Buszydlik, Karol Dobiczek, Arie van Deursen, and Cynthia CS Liem. 2023. “Endogenous Macrodynamics in Algorithmic Recourse.” In *2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)*, 418–31. IEEE.

Antorán, Javier, Umang Bhatt, Tameem Adel, Adrian Weller, and José Miguel Hernández-Lobato. 2020. “Getting a Clue: A Method for Explaining Uncertainty Estimates.” <https://arxiv.org/abs/2006.06848>.

Joshi, Shalmali, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh. 2019. “Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems.” <https://arxiv.org/abs/1907.09615>.

Kaggle. 2011. “Give Me Some Credit, Improve on the State of the Art in Credit Scoring by Predicting the Probability That Somebody Will Experience Financial Distress in the Next Two Years.” https://www.kaggle.com/c/GiveMeSomeCredit; Kaggle. <https://www.kaggle.com/c/GiveMeSomeCredit>.

Laugel, Thibault, Marie-Jeanne Lesot, Christophe Marsala, Xavier Renard, and Marcin Detyniecki. 2017. “Inverse Classification for Comparison-Based Interpretability in Machine Learning.” <https://arxiv.org/abs/1712.08443>.

Mothilal, Ramaravind K, Amit Sharma, and Chenhao Tan. 2020. “Explaining Machine Learning Classifiers Through Diverse Counterfactual Explanations.” In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607–17. <https://doi.org/10.1145/3351095.3372850>.

Pawelczyk, Martin, Teresa Datta, Johannes van-den-Heuvel, Gjergji Kasneci, and Himabindu Lakkaraju. 2023. “Probabilistically Robust Recourse: Navigating the Trade-Offs Between Costs and Robustness in Algorithmic Recourse.” <https://arxiv.org/abs/2203.06768>.

Schut, Lisa, Oscar Key, Rory Mc Grath, Luca Costabello, Bogdan Sacaleanu, Yarin Gal, et al. 2021. “Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties.” In *International Conference on Artificial Intelligence and Statistics*, 1756–64. PMLR.

Tolomei, Gabriele, Fabrizio Silvestri, Andrew Haines, and Mounia Lalmas. 2017. “Interpretable Predictions of Tree-Based Ensembles via Actionable Feature Tweaking.” In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 465–74. <https://doi.org/10.1145/3097983.3098039>.

Wachter, Sandra, Brent Mittelstadt, and Chris Russell. 2017. “Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR.” *Harv. JL & Tech.* 31: 841. <https://doi.org/10.2139/ssrn.3063289>.
