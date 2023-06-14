# Report: Integrating Python and R models into the package

This report describes the design choices we made throughout the process of integrating models defined using PyTorch and Torch for R into the `CounterfactualExplanations.jl` package.

The end result was functionality that supports generating counterfactuals for any neural network the user has defined and trained in Python using PyTorch, regardless of the specific architectural details of the model. We also implemented the basis for adding support for models defined using R, but did not make this feature available for the end user yet, for reasons that will be discussed below.

This report will be structured as follows: first, we will discuss the most important problems that we ran into while implementing support for Python and R models, and second, we will provide an overview of the design choices made.

## Problems

### `ONNX.jl` is not sufficiently advanced to handle the conversion

When starting out working on this issue, we identified the `ONNX.jl` library as the one best suited for handling the conversion between Python or R and Julia in close communication with the client. `ONNX.jl` is a library for transfering external models into Julia by reading the computational graphs of pretrained models from ONNX format to `Flux.jl`. This would have given `Flux.jl`, a package already heavily integrated into our package, gradient access to any model defined using PyTorch or Torch for R, which would have enabled generating counterfactual explanations for Python and R models in a very similar way to how the package already generates explanations for models from `Flux.jl`. This solution also looked like the best one because it would have enabled using a single library for conversion from both Python and R, avoiding problems with conflicting Torch sessions that will be described at length below.

Unfortunately, we found out during a closer inspection that `ONNX.jl` does not suit our purposes well enough and had to resort to alternative solution. We decided not to use `ONNX.jl` for the following reasons:
- As of 06/06/2023, the `ONNX.jl` library is [in the process of total reconstruction](https://github.com/FluxML/ONNX.jl). As noted in [JuliaHub](https://juliahub.com/ui/Packages/ONNX/QUmGg/0.2.4) and confirmed by our inspection of the package, no conversion from the Torch computational graph to `Flux.jl` computational graph has been implemented yet. As implementing this conversion would be a project worthy of a whole software project and the package would not have been useful for us without this conversion, we decided that it is best to use another solution that is more mature.
- The library is also not very actively maintained. When we started working on the issue, the last release of the package had been made on [June 18th, 2022](https://github.com/FluxML/ONNX.jl/releases). Though the developers of the package had been mentioning progress on the reconstruction efforts and plans to provide proper documentation soon already in [January 2022](https://github.com/FluxML/ONNX.jl/issues/60), it seemed like no substantial progress had been made after that and the documentation was still completely missing. Though a new release was made on March 26th, 2023, this release did not complete the reconstruction process.

For all these reasons, we decided to use alternative options for solving this issue: `PythonCall.jl` and `RCall.jl`. These choices will be discussed below.

### The tests for the Python conversion fail on the Ubuntu Julia 1.7 and OSX pipelines

Once we had finished the implementation of the Python models, we noticed that though the pipeline was passing on our machines as well as on most of the GitHub virtual machines we ran it on, some pipelines on the remote were failing: the pipeline testing our code on Ubuntu using Julia 1.7, and the pipeline testing it on OSX machines. The reasons behind the failures were different: 

### `PythonCall.jl` and `RCall.jl` cannot be used together in the same session

## Design choices

### Using `PythonCall.jl`

When we started looking into possible ways of generating counterfactuals for Python models using Julia, we found three libraries that offered functionalities for conversion between Python and Julia:
- `ONNX.jl`
    - This is a library for transfering external models into Julia by reading the computational graphs of pretrained models from ONNX format to `Flux.jl`. This would have given `Flux.jl`, a package already heavily integrated into our package, gradient access to any model defined using PyTorch, which would have enabled generating counterfactual explanations for PyTorch models in a very similar way to how the package already generates explanations for models from `Flux.jl`.
    - Unfortunately, as of 06/06/2023, the `ONNX.jl` library is [in the process of total reconstruction](https://github.com/FluxML/ONNX.jl). As noted in [JuliaHub](https://juliahub.com/ui/Packages/ONNX/QUmGg/0.2.4), no conversion to `Flux.jl` is implemented yet. As implementing this conversion would be a project worthy of a whole software project, we quickly realized that `ONNX.jl` is not a feasible option for implementing support for Python models.
    - Since `ONNX.jl` has not been maintained particularly actively in recent times, we decided that it's best to look for an alternative solution.
- `PyCall.jl`
    - This is a library for translating Python code into Julia code and vice versa.
    - The client had already tried implementing support for Python models using this library, but ran into various problems: using both PyTorch through `PyCall.jl` and torch for R through `RCall.jl` in the same Julia session caused Julia to crash. We looked into the problems a bit and decided that there's a better alternative to `PyCall.jl` that works in a different way and avoids this problem.
- `PythonCall.jl`
    - This is also a library for translating Python code into Julia code and vice versa.
    - Compared to `PyCall.jl`, this library is more recent and more actively maintained. For this reason, we decided that it would be easiest to solve the problems the client had been running into using this library.

For the reasons described above, we decided to solve this problem using `PythonCall.jl`.

### Supporting only pretrained PyTorch models

Once we had implemented basic support for generating counterfactuals for PyTorch models through `PythonCall.jl`, we discussed with the client whether the package should offer support only for PyTorch models predefined by the user in a Python environment or whether we should also offer support for training PyTorch models inside the package. Since defining PyTorch models through Julia code is tedious and error-prone, the client expected most users to only want counterfactuals for models they have themselves previously trained using PyTorch. Since the client has considerable machine learning experience and offered compelling reasons for this design choice, we decided not to offer special support for defining and training PyTorch models through our package.