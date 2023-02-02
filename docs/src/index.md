
``` @meta
CurrentModule = CounterfactualExplanations
```

![](assets/wide_logo.png)

Documentation for [CounterfactualExplanations.jl](https://github.com/pat-alt/CounterfactualExplanations.jl).

``` @meta
CurrentModule = CounterfactualExplanations 
```

`CounterfactualExplanations.jl` is a package for generating Counterfactual Explanations (CE) and Algorithmic Recourse (AR) for black-box algorithms. Both CE and AR are related tools for explainable artificial intelligence (XAI). While the package is written purely in Julia, it can be used to explain machine learning algorithms developed and trained in other popular programming languages like Python and R. See below for a short introduction and other resources or dive straight into the [docs](https://pat-alt.github.io/CounterfactualExplanations.jl/dev).

## Installation 🚩

You can install the stable release from [Julia’s General Registry](https://github.com/JuliaRegistries/General) as follows:

``` julia
using Pkg
Pkg.add("CounterfactualExplanations")
```

`CounterfactualExplanations.jl` is under active development. To install the development version of the package you can run the following command:

``` julia
using Pkg
Pkg.add(url="https://github.com/pat-alt/CounterfactualExplanations.jl")
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

Consider the following real-world scenario: a retail bank is using a black-box model trained on their clients’ credit history to decide whether they will provide credit to new applicants. To simulate this scenario, we have pre-trained a binary classifier on the publically available Give Me Some Credit dataset that ships with this package (Kaggle 2011).

The figure below shows counterfactuals for 10 randomly chosen individuals that would have been denied credit initially.

![](index_files/figure-commonmark/cell-5-output-1.svg)

### Example: MNIST

The figure below shows a counterfactual generated for an image classifier trained on MNIST: in particular, it demonstrates which pixels need to change in order for the classifier to predict 4 instead of 9.

The counterfactual was produced using REVISE (Joshi et al. 2019):

``` julia
# Define generator:
generator = REVISEGenerator(
  opt = Descent(0.1),
  decision_threshold = 0.95,
  λ = 0.01
)
# Generate recourse:
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
```

![](index_files/figure-commonmark/cell-10-output-1.svg)

## 🔍 Usage example

Generating counterfactuals will typically look like follows.

We have some pre-trained model that was fitted to data:

``` julia
# Data and Classifier:
counterfactual_data = load_linearly_separable(1000)
M = fit_model(counterfactual_data, :Linear)
```

For some individual and target outcome, we want to understand what a valid counterfactual in the target class looks like:

``` julia
# Randomly selected factual:
x = select_factual(counterfactual_data,rand(1:size(counterfactual_data.X,2)))
y = predict_label(M, counterfactual_data, x)[1]
target = counterfactual_data.y_levels[counterfactual_data.y_levels .!= y][1]
```

To this end, we specify a counterfactual generator of our choice:

``` julia
# Counterfactual search:
generator = DiCEGenerator(
  opt = Descent(0.01)
)
```

Here, we have chosen to use the `DiCEGenerator` to move the individual from its factual label 2 to the target label 1.

With all of our ingredients specified, we finally generate counterfactuals using a simple API call:

``` julia
ce = generate_counterfactual(
  x, target, counterfactual_data, M, generator; 
  num_counterfactuals=3
)
```

The animation below shows the resulting counterfactual path:

![](www/intro.gif)

![](README_files/intro.gif)

## Implemented Counterfactual Generators:

Currently, the following counterfactual generators are implemented:

- Generic (Wachter, Mittelstadt, and Russell 2017)
- Greedy (Schut et al. 2021)
- DiCE (Mothilal, Sharma, and Tan 2020)
- Latent Space Search as in REVISE (Joshi et al. 2019) and CLUE (Antorán et al. 2020)
- ClaPROAR (Altmeyer et al. 2023)
- GravitationalGenerator (Altmeyer et al. 2023)

## 🎯 Goals and limitations

The goal of this library is to contribute to efforts towards trustworthy machine learning in Julia. The Julia language has an edge when it comes to trustworthiness: it is very transparent. Packages like this one are generally written in pure Julia, which makes it easy for users and developers to understand and contribute to open-source code. Eventually, this project aims to offer a one-stop-shop of counterfactual explanations. We want to deliver a package that is at least at par with the [CARLA](https://github.com/carla-recourse/CARLA) Python library in terms of its functionality. Currently, the package falls short of this goal in some ways:

1.  The number of counterfactual generators is still limited.
2.  Mutability constraints are still not supported for Latent Space generators.

Additionally, our ambition is to enhance the package through the following features:

4.  Language interoperability with Python and R: currently still only experimental.
5.  Support for machine learning models trained in [`MLJ.jl`](https://alan-turing-institute.github.io/MLJ.jl/dev/).
6.  Additional datasets for testing, evaluation and benchmarking.
7.  Support for regression models.

## 🛠 Contribute

Contributions of any kind are very much welcome! Take a look at the [issue](https://github.com/pat-alt/CounterfactualExplanations.jl/issues) to see what things we are currently working on.

If any of the below applies to you, this might be the right open-source project for you:

- You’re an expert in Counterfactual Explanations or Explainable AI more broadly and you are curious about Julia.
- You’re experienced with Julia and are happy to help someone less experienced to up their game. Ideally, you are also curious about Trustworthy AI.
- You’re new to Julia and open-source development and would like to start your learning journey by contributing to a recent and active development. Ideally, you are familiar with machine learning.

[@pat-alt](https://github.com/pat-alt) here: I am still very much at the beginning of my Julia journey, so if you spot any issues or have any suggestions for design improvement, please just open [issue](https://github.com/pat-alt/CounterfactualExplanations.jl/issues) or start a [discussion](https://github.com/pat-alt/CounterfactualExplanations.jl/discussions). Our goal is to provide a go-to place for counterfactual explanations in Julia.

For more details on how to contribute see [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/contributing/). Please follow the [SciML ColPrac guide](https://github.com/SciML/ColPrac).

## Citation 🎓

If you want to use this codebase, please consider citing:

    @software{altmeyer2022CounterfactualExplanations,
      author = {Patrick Altmeyer},
      title = {{CounterfactualExplanations.jl - a Julia package for Counterfactual Explanations and Algorithmic Recourse}},
      url = {https://github.com/pat-alt/CounterfactualExplanations.jl},
      year = {2022}
    }

## References 📚

Altmeyer, Patrick, Giovan Angela, Aleksander Buszydlik, Karol Dobiczek, Arie van Deursen, and Cynthia Liem. 2023. “Endogenous Macrodynamics in Algorithmic Recourse.” In *First IEEE Conference on Secure and Trustworthy Machine Learning*.

Antorán, Javier, Umang Bhatt, Tameem Adel, Adrian Weller, and José Miguel Hernández-Lobato. 2020. “Getting a Clue: A Method for Explaining Uncertainty Estimates.” <https://arxiv.org/abs/2006.06848>.

Joshi, Shalmali, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh. 2019. “Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems.” <https://arxiv.org/abs/1907.09615>.

Kaggle. 2011. “Give Me Some Credit, Improve on the State of the Art in Credit Scoring by Predicting the Probability That Somebody Will Experience Financial Distress in the Next Two Years.” Kaggle. <https://www.kaggle.com/c/GiveMeSomeCredit>.

Mothilal, Ramaravind K, Amit Sharma, and Chenhao Tan. 2020. “Explaining Machine Learning Classifiers Through Diverse Counterfactual Explanations.” In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607–17.

Schut, Lisa, Oscar Key, Rory Mc Grath, Luca Costabello, Bogdan Sacaleanu, Yarin Gal, et al. 2021. “Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties.” In *International Conference on Artificial Intelligence and Statistics*, 1756–64. PMLR.

Wachter, Sandra, Brent Mittelstadt, and Chris Russell. 2017. “Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR.” *Harv. JL & Tech.* 31: 841.
