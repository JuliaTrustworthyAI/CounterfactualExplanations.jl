# CounterfactualExplanations

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/CounterfactualExplanations.jl/dev)
[![Build Status](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/CounterfactualExplanations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/CounterfactualExplanations.jl)

CounterfactualExplanations.jl is a package for generating Counterfactual Explanations (CE) and Algorithmic Recourse (AR) in Julia Language. Both CE and AR are related tools for interpretable machine learning. All too often human operators rely blindly on decisions made by black-box algorithms. Counterfactual Explanations can help programmers make sense of the systems they build: they explain how inputs into a system need to change for it to produce a different output. CEs that involve realistic and actionable changes can be used for the purpose of individual recourse: Algorithmic Recourse (AR) offers individuals subject to algorithms a way to turn a negative decision into positive one.

## Installation

The package is in its early stages of development and not yet registered. In the meantime it can be installed as follows:

```julia
using Pkg
Pkg.add("https://github.com/pat-alt/CounterfactualExplanations.jl")
```

To instead install the development version of the package you can run the following command:

```julia
using Pkg
Pkg.add(url="https://github.com/pat-alt/CounterfactualExplanations.jl", rev="dev")
```

