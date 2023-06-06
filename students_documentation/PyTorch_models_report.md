# Report: Integrating PyTorch models into the package

This report describes the design choices we made throughout the process of integrating models defined using PyTorch into the `CounterfactualExplanations.jl` package. We wrote functionality that supports generating counterfactuals for any neural network the user has defined and trained in PyTorch, regardless of the specific architectural details of the model.

## Design choice 1: using PythonCall.jl

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

## Design choice 2: Supporting only pretrained PyTorch models

Once we had implemented basic support for generating counterfactuals for PyTorch models through `PythonCall.jl`, we discussed with the client whether the package should offer support only for PyTorch models predefined by the user in a Python environment or whether we should also offer support for training PyTorch models inside the package. Since defining PyTorch models through Julia code is tedious and error-prone, the client expected most users to only want counterfactuals for models they have themselves previously trained using PyTorch. Since the client has considerable machine learning experience and offered compelling reasons for this design choice, we decided not to offer special support for defining and training PyTorch models through our package.