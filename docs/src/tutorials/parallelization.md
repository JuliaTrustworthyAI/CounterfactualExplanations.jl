# Parallelization

``` @meta
CurrentModule = CounterfactualExplanations 
```

Version `0.1.15` adds support for parallelization through multi-processing. Currently, the only available backend for parallelization is [MPI.jl](https://juliaparallel.org/MPI.jl/latest/).

## Available functions

Parallelization is only available for certain functions. To check if a function is parallelizable, you can use `parallelizable` function:

``` julia
using CounterfactualExplanations.Evaluation: evaluate, benchmark
println(parallelizable(generate_counterfactual))
println(parallelizable(evaluate))
println(parallelizable(predict_label))
```

    true
    true
    false

In the following, we will generate multiple counterfactuals and evaluate them in parallel:

``` julia
using CounterfactualExplanations.Parallelization
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual), 10)
xs = select_factual(counterfactual_data, chosen)
```

## MPI

!!! note
    To use MPI, you need to have MPI installed on your machine. Running the following code straight from a running Julia session will work if you have MPI installed on your machine, but it will be run on a single process. To run the code in parallel, you need to run it from the command line with `mpirun` or `mpiexec`. For example, to run the code on 4 processes, you can run the following command from the command line. For more information, see [MPI.jl](https://juliaparallel.org/MPI.jl/latest/). 

We first instantiate an `MPIParallelizer` object:

``` julia
parallelizer = MPIParallelizer()
```

    MPIParallelizer()

To generate counterfactuals in parallel, we use the `parallelize` function:

``` julia
ces = parallelize(
    parallelizer,
    generate_counterfactual,
    xs,
    target,
    counterfactual_data,
    M,
    generator
)
```

    [ Info: Using `MPI.jl` for multi-processing.

    10-element Vector{CounterfactualExplanation}:
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     CounterfactualExplanation
    Convergence: ✅ after 9 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.

To evaluate counterfactuals in parallel, we again use the `parallelize` function:

``` julia
parallelize(
    parallelizer,
    evaluate,
    ces;
    report_meta = true
)
```

    [ Info: Using `MPI.jl` for multi-processing.

Benchmarks can also be run with parallelization by specifying `parallelizer` argument:

``` julia
# Models:
bmk = benchmark(counterfactual_data; parallelizer = parallelizer)
```
