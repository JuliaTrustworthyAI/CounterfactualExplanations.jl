# CounterfactualExplanations

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/dev) [![Build Status](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl)

`CounterfactualExplanations.jl` is a package for generating Counterfactual Explanations (CE) and Algorithmic Recourse (AR) for black-box algorithms. Both CE and AR are related tools for interpretable machine learning. While the package is written purely in Julia, it can be used to explain machine learning algorithms developed and trained in other popular programming languages like Python and R. See below for short introduction and other resources or dive straight into the [docs](https://pat-alt.github.io/CounterfactualExplanations.jl/dev).

## Installation

The first release of this package is now on [Julia’s General Registry](https://github.com/JuliaRegistries/General) and can be installed as follows:

``` julia
using Pkg
Pkg.add("CounterfactualExplanations")
```

CounterfactualExplanations.jl is currently under active development. To install the development version of the package you can run the following command:

``` julia
using Pkg
Pkg.add(url="https://github.com/pat-alt/CounterfactualExplanations.jl", rev="dev")
```

## Background and motivation

Algorithms used for automated decision-making such as deep neural networks have become so complex and opaque over recent years that they are generally considered as black boxes. This creates the following undesirable scenario: the human operators in charge of the black-box decision-making system do not understand how it works and essentially rely on it blindly. Conversely, those individuals who are subject to the decisions produced by such systems typically have no way of challenging them.

> “You cannot appeal to (algorithms). They do not listen. Nor do they bend.”
>
> — Cathy O’Neil in [*Weapons of Math Destruction*](https://en.wikipedia.org/wiki/Weapons_of_Math_Destruction), 2016

**Counterfactual Explanations can help programmers make sense of the systems they build: they explain how inputs into a system need to change for it to produce a different output**. The figure below, for example, shows various counterfactuals generated through different approaches that all turn the predicted label of some classifier from a 9 into a 4. CEs that involve realistic and actionable changes such as the one on the far right can be used for the purpose of individual counterfactual.

![Realistic counterfactual explanations for MNIST data: turning a 4 into a 9.](https://raw.githubusercontent.com/pat-alt/CounterfactualExplanations.jl/main/docs/src/examples/image/www/MNIST_9to4.png)

**Algorithmic Recourse (AR) offers individuals subject to algorithms a way to turn a negative decision into positive one**. The figure below illustrates the point of AR through a toy example: it shows the counterfactual path of one sad cat 🐱 that would like to be grouped with her cool dog friends. Unfortunately, based on her tail length and height she was classified as a cat by a black-box classifier. The recourse algorithm perturbs her features in such a way that she ends up crossing the decision boundary into a dense region inside the target class.

![A sad 🐱 on its counterfactual path to its cool dog friends.](https://raw.githubusercontent.com/pat-alt/CounterfactualExplanations.jl/main/docs/src/www/recourse_laplace.gif)

## Usage example

Generating counterfactuals will typically look like follows:

``` julia
using CounterfactualExplanations

# Raw Data:
using CounterfactualExplanations.Data: cats_dogs_data
x, y = cats_dogs_data()

# Data preprocessing:
counterfactual_data = CounterfactualData(x,y)

# Model (pre-trained):
using CounterfactualExplanations.Data: cats_dogs_laplace
import CounterfactualExplanations.Models: probs
la = cats_dogs_laplace()

# Counterfactual search:
x = select_factual(counterfactual_data, 1) # factual
target = round(probs(la, x)) == 1.0 ? 0.0 : 1.0
generator = GenericGenerator()
counterfactual = generate_counterfactual(x, target, counterfactual_data, la, generator)
```

## Goals and limitations

The goal for this library is to contribute to efforts towards trustworthy machine learning in Julia. The Julia language has an edge when it comes to trustworthiness: it is very transparent. Packages like this one are generally written in pure Julia, which makes it easy for users and developers to understand and contribute to open source code. Eventually the aim for this project is to offer a one-stop-shop of counterfactual explanations. We want to deliver a package that is at least at par with the [CARLA](https://github.com/carla-recourse/CARLA) Python 🐍 library in terms of its functionality. Contrary to CARLA, we aim for languague interoperability. Currently the package falls short of this goal in a number of ways: 1) the number of counterfactual generators is limited, 2) the data preprocessing functionality needs to be extended, 3) it has not yet gone through a formal review.

## Contribute

`CounterfactualExplanations.jl` is designed to be extensible: through multiple dispatch and modularization we hope to make it as straight-forward as possible for members of the community to contribute to its functionality. At the moment we are primarily looking for the following contributions:

1.  Additional counterfactual generators.
2.  Additional predictive models.
3.  More examples to be added to the documentation.
4.  Native support for categorical features.
5.  Support for regression models.

For more details on how to contribute see [here](https://www.paltmeyer.com/CounterfactualExplanations.jl/dev/contributing/).

## Citation

If you want to use this codebase, please cite:

    @software{altmeyer2022CounterfactualExplanations,
      author = {Patrick Altmeyer},
      title = {{CounterfactualExplanations.jl - a Julia package for Counterfactual Explanations and Algorithmic Recourse}},
      url = {https://github.com/pat-alt/CounterfactualExplanations.jl},
      version = {0.1.0},
      year = {2022}
    }
