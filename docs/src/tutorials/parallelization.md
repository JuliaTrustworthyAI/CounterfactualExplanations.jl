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
    To use MPI, you need to have MPI installed on your machine. Running the following code straight from a running Julia session will work if you have MPI installed on your machine, but it will be run on a single process. To execute the code on multiple processes, you need to run it from the command line with `mpirun` or `mpiexec`. For example, to run a script on 4 processes, you can run the following command from the command line:
    
    ```


    mpiexecjl --project -n 4 julia -e 'include("docs/src/srcipts/mpi.jl")'
    ```

    For more information, see [MPI.jl](https://juliaparallel.org/MPI.jl/latest/). 

We first instantiate an `MPIParallelizer` object:

``` julia
import MPI
MPI.Init()
parallelizer = MPIParallelizer(MPI.COMM_WORLD)
```

    Running on 1 processes.

    [ Info: Using `MPI.jl` for multi-processing.

    MPIParallelizer(MPI.Comm(1140850688), 0, 1)

To generate counterfactuals in parallel, we use the `parallelize` function:

``` julia
ces = @with_parallelizer parallelizer begin
    generate_counterfactual(
        xs,
        target,
        counterfactual_data,
        M,
        generator
    )
end
```

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
@with_parallelizer parallelizer evaluate(ces; report_meta = true)
```

!!! tip
    Note that parallelizable processes can be supplied as input to the macro either as a block or directly as an expression.

Benchmarks can also be run with parallelization by specifying `parallelizer` argument:

``` julia
# Models:
bmk = benchmark(counterfactual_data; parallelizer = parallelizer)
```

The following code snippet shows a complete example script that uses MPI for running a benchmark in parallel:

``` julia
using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Evaluation: benchmark
using CounterfactualExplanations.Models
using CounterfactualExplanations.Parallelization
import MPI

MPI.Init()

counterfactual_data = load_linearly_separable()
M = fit_model(counterfactual_data, :Linear)
factual = 1
target = 2
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual), 100)
xs = select_factual(counterfactual_data, chosen)
generator = GenericGenerator()

parallelizer = MPIParallelizer(MPI.COMM_WORLD)

bmk = benchmark(counterfactual_data; parallelizer=parallelizer)

MPI.Finalize()
```

The file can be executed from the command line as follows:

    mpiexecjl --project -n 4 julia -e 'include("docs/src/srcipts/mpi.jl")'
