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

### The tests for the Python conversion fail when using Julia 1.7

Once we had finished the implementation of the Python models, we noticed that the pipeline on the remote testing our code on Ubuntu using Julia 1.7 was failing. Though all of the tests were passing, there seemed to be a permission error during clean-up after the tests. We investigated the problem with a testing expert from the client's team, [Antony Bartlett](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/multimedia-computing/people/antony-bartlett), but could still not find a good solution for multiple days. Eventually, we concluded in a discussion with the client, Antony Bartlett and ourselves that our time would be better spent focusing on other issues and that we can just note in the documentation of our package that the functionality of creating counterfactuals for PyTorch models is only available for Julia versions 1.8 and above. Since the issue did not seem to have good solutions, even after exploring the problem with an experienced testing expert for multiple days, since Julia 1.7 is nowadays used significantly less than Julia 1.8 or 1.9, and since this resolution was fine for the client, we agreed with this suggestion and made the Python-related functionality unavailable for people using Julia 1.7.

### `PythonCall.jl` and `RCall.jl` cannot be used together in the same session

The third important problem we ran into while implementing this functionality was that it turned out to be impossible to use PyTorch and Torch for R during a single session: trying to do this just made the session crash. The reason for that appeared to be that the Torch for R implementation is [based on PyTorch](https://torch.mlverse.org/) and this effectively meant that we were trying to initialize two conflicting Torch sessions within the same Julia session. This issue would have been avoided had we been able to use `ONNX.jl`, but for the reasons explained above, that was not an option.

We discussed the problem with the client and decided based on both our own exploration of the problem and [the client's previous experience trying to get `PyCall.jl` and `RCall.jl` to work together](https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl/pull/32) that resolving this issue would take too much time and is thus outside of the scope of the software project.

Due to the fact that the package crashes when trying to load PyTorch and R models within the same session, they were also not testable during the same session. For this reason, we considered simply leaving the functionality for generating counterfactuals for R models out of the package, since the client found support for Python models more important. However, we already implemented the functionality for R models at the point where we were considering this, and the client told us he would prefer to have the logic for R models in the package so that he would not need to implement support for R models from scratch in the future when trying to resolve the problem of incompatible Torch sessions on his own in the future. This reasoning made sense to us, so we left untested functionality for R models inside the package, but made it inaccessible to users.


## Design choices

### Using `PythonCall.jl` and `RCall.jl`

When looking into alternatives to `ONNX.jl`, we found that a natural solution for R models would be translating the R code into Julia code using `RCall.jl` and then generating counterfactuals as we usually do. For Python models, however, we found two libraries that both offered functionalities for conversion between Python and Julia:
- `PyCall.jl`
    - This is a library for translating Python code into Julia code and vice versa.
    - The client had already tried implementing support for Python models using this library, but ran into various problems: for example, it was very difficult to properly handle Python dependencies through `PyCall.jl`. We looked into the problems a bit and decided that there's a better alternative to `PyCall.jl` that works in a different way and avoids this problem.
- `PythonCall.jl`
    - This is also a library for translating Python code into Julia code and vice versa.
    - Compared to `PyCall.jl`, this library is more recent and more actively maintained. Furthermore, it offers better dependency management functionalities for Python packages through `CondaPkg.jl`. For this reason, we decided that it would be easiest to solve the problems the client had been running into using this library.

For the reasons described above, we decided to use `RCall.jl` and `PythonCall.jl`.

### Supporting only pretrained PyTorch models

Once we had implemented basic support for generating counterfactuals for PyTorch models through `PythonCall.jl`, we discussed with the client whether the package should offer support only for PyTorch models predefined by the user in a Python environment or whether we should also offer support for training PyTorch models inside the package. Since defining PyTorch models through Julia code is tedious and error-prone, the client expected most users to only want counterfactuals for models they have themselves previously trained using PyTorch. Since the client has considerable machine learning experience and offered compelling reasons for this design choice, we decided not to offer special support for defining and training PyTorch models through our package.