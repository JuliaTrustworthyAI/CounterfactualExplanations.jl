# AlgorithmicRecourse

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pat-alt.github.io/AlgorithmicRecourse.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pat-alt.github.io/AlgorithmicRecourse.jl/dev)
[![Build Status](https://github.com/pat-alt/AlgorithmicRecourse.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pat-alt/AlgorithmicRecourse.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/pat-alt/AlgorithmicRecourse.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pat-alt/AlgorithmicRecourse.jl)

A package for counterfactual explanations (CE) and algorithmic recourse (AR) in Julia. The former (CE) is a common approach towards explaining machine learning models. The latter (AR) uses counterfactual explanations to systematically provide recourse to individuals faced with an undesirable algorithmic outcome. 

## Installation

The package is in its early stages of development and not yet registered. In the meantime it can be installed as follows:

```julia
using Pkg
Pkg.add("https://github.com/pat-alt/AlgorithmicRecourse.jl")
```

To instead install the development version of the package you can run the following command:

```julia
using Pkg
Pkg.add(url="https://github.com/pat-alt/AlgorithmicRecourse.jl", rev="dev")
```

