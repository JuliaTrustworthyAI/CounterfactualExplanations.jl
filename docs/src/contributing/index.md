```@meta
CurrentModule = CounterfactualExplanations 
```

# Contributing

`CounterfactualExplanations.jl` is designed to be scalable: through multiple dispatch and modularization we hope to make it as straight-forward as possible for members of the community to contribute to its functionality. At the moment we are primarily looking for the following contributions:

1. Additional counterfactual generators.
2. Additional predictive models.
3. More examples to be added to the documentation.
4. Native support for categorical features.
5. Support for regression models.

We are also interested in suggestions on how to adjust and improve the inner workings of our package. To facilitate this process the following page explain and justify the package architecture and the design choices we have made. 

## How to contribute?

All of the following contributions are welcome:

1. Should you spot any errors or something is not working, please just open an [issue](https://github.com/pat-alt/CounterfactualExplanations.jl/issues).
2. If you want to contribute your own code, please proceed as follows:
   - Fork this repo and clone your fork: `git clone https://github.com/your_username/CounterfactualExplanations.jl`.
   - Add a remote corresponding to this repository: `git remote add upstream https://github.com/pat-alt/CounterfactualExplanations.jl.git`
   - Implement your modifications and submit a pull request.
3. For any other questions or comments you can also start a [discussion](https://github.com/pat-alt/CounterfactualExplanations.jl/discussions).