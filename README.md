# CounterfactualExplanations

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/stable) -->

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/dev) [![Build Status](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl)

CounterfactualExplanations.jl is a Julia package for generating Counterfactual Explanations (CE) and Algorithmic Recourse (AR) for black-box algorithms. Both CE and AR are related tools for interpretable machine learning. See below for short introduction and other resources or dive straight into the [docs](https://pat-alt.github.io/CounterfactualExplanations.jl/dev).

## Installation

The package is in its early stages of development and currently awaiting registry on [Julia’s General Registry](https://github.com/JuliaRegistries/General). In the meantime it can be installed as follows:

``` julia
using Pkg
Pkg.add("https://github.com/pat-alt/CounterfactualExplanations.jl")
```

To instead install the development version of the package you can run the following command:

``` julia
using Pkg
Pkg.add(url="https://github.com/pat-alt/CounterfactualExplanations.jl", rev="dev")
```

## Background and motivation

Algorithms used for automated decision-making such as deep neural networks have become so complex and opaque over recent years that they are generally considered as black boxes. This creates the following undesirable scenario: the human operators in charge of the black-box decision-making system do not understand how it works and essentially rely on it blindly. Conversely, those individuals who are subject to the decisions produced by such systems typically have no way of challenging them.

> “You cannot appeal to (algorithms). They do not listen. Nor do they bend.”
>
> — Cathy O’Neil in [*Weapons of Math Destruction*](https://en.wikipedia.org/wiki/Weapons_of_Math_Destruction), 2016

**Counterfactual Explanations can help programmers make sense of the systems they build: they explain how inputs into a system need to change for it to produce a different output**. [Figure 1](#fig-mnist), for example, shows various counterfactuals generated through different approaches that all turn the predicted label of some classifier from a 9 into a 4. CEs that involve realistic and actionable changes such as the one on the far right can be used for the purpose of individual recourse.

<figure>
<img src="https://raw.githubusercontent.com/pat-alt/CounterfactualExplanations.jl/main/docs/src/examples/image/www/MNIST_9to4.png" id="fig-mnist" alt="Figure 1: Realistic counterfactual explanations for MNIST data: turning a 4 into a 9." />
<figcaption aria-hidden="true">Figure 1: Realistic counterfactual explanations for MNIST data: turning a 4 into a 9.</figcaption>
</figure>

**Algorithmic Recourse (AR) offers individuals subject to algorithms a way to turn a negative decision into positive one**. [Figure 2](#fig-cat) illustrates the point of AR through a toy example: it shows the counterfactual path of one sad cat 🐱 that would like to be grouped with her cool dog friends. Unfortunately, based on her tail length and height she was classified as a cat by a black-box classifier. The recourse algorithm perturbs her features in such a way that she ends up crossing the decision boundary into a dense region inside the target class.

<figure>
<img src="https://raw.githubusercontent.com/pat-alt/CounterfactualExplanations.jl/main/docs/src/examples/www/recourse_laplace.gif" id="fig-cat" alt="Figure 2: A sad 🐱 on its counterfactual path to its cool dog friends." />
<figcaption aria-hidden="true">Figure 2: A sad 🐱 on its counterfactual path to its cool dog friends.</figcaption>
</figure>

## Usage example

Generating counterfactuals will typically look like follows:

``` julia
using CounterfactualExplanations
using CounterfactualExplanations.Models
w = [1.0 -2.0] # true coefficients
b = [0]
𝑴 = LogisticModel(w, b)
x̅ = [-1,0.5]
target = 1.0
γ = 0.9
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ); # generate recourse
```

We can see that the counterfactual label y̲ corresponds to the target:

``` julia
julia> recourse.y̲
1.0

julia> recourse.x̅
2-element Vector{Float64}:
 -1.0
  0.5

julia> recourse.y̅
0.0

julia> recourse.x̲
2-element Vector{Float64}:
 -0.15867040347424893
 -1.182659193051502

julia> recourse.y̲
1.0
```

## Goals and limitations

The goal for this library is to contribute to efforts towards trustworthy machine learning in Julia. The Julia language has an edge when it comes to trustworthiness: it is very transparent. Packages like this one are generally written in 100% Julia, which makes it easy for users and developers to understand and contribute to open source code.

Eventually the aim for this project is to be at least at par with the amazing [CARLA](https://github.com/carla-recourse/CARLA) Python library which was presented at NeurIPS 2021. Currently CounterfactualExplanations.jl falls short of this goal in a number of ways: 1) the number of counterfactual generators is limited, 2) it lacks a framework for evaluating and benchmarking different generators, 3) it has so far been a one-person effort and not yet gone through a formal review.

## Citation

If you want to use this codebase, please cite:

    @software{altmeyer2022CounterfactualExplanations,
      author = {Patrick Altmeyer},
      title = {{CounterfactualExplanations.jl - a Julia package for Counterfactual Explanations and Algorithmic Recourse}},
      url = {https://github.com/pat-alt/CounterfactualExplanations.jl},
      version = {0.1.0},
      year = {2022}
    }
