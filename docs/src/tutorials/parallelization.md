

``` @meta
CurrentModule = CounterfactualExplanations 
```

# Parallelization

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
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual), 1000)
xs = select_factual(counterfactual_data, chosen)
```

## Multi-threading

We first instantiate an `ThreadParallelizer` object:

``` julia
parallelizer = ThreadsParallelizer()
```

    ThreadsParallelizer()

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

    Generating counterfactuals ...   0%|       |  ETA: 0:01:32 (92.31 ms/it)Generating counterfactuals ... 100%|███████| Time: 0:00:01 ( 1.64 ms/it)

    1000-element Vector{AbstractCounterfactualExplanation}:
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     ⋮
     CounterfactualExplanation
    Convergence: ✅ after 9 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.

To evaluate counterfactuals in parallel, we again use the `parallelize` function:

``` julia
@with_parallelizer parallelizer evaluate(ces)
```

    Evaluating counterfactuals ...   0%|       |  ETA: 0:07:27 ( 0.45  s/it)Evaluating counterfactuals ... 100%|███████| Time: 0:00:00 ( 0.90 ms/it)

    1000-element Vector{Any}:
     Vector[[1.0], Float32[3.2939816], [0.0]]
     Vector[[1.0], Float32[3.019046], [0.0]]
     Vector[[1.0], Float32[3.701171], [0.0]]
     Vector[[1.0], Float32[2.5611918], [0.0]]
     Vector[[1.0], Float32[2.9027307], [0.0]]
     Vector[[1.0], Float32[3.7893882], [0.0]]
     Vector[[1.0], Float32[3.5026522], [0.0]]
     Vector[[1.0], Float32[3.6317568], [0.0]]
     Vector[[1.0], Float32[3.084984], [0.0]]
     Vector[[1.0], Float32[3.2268934], [0.0]]
     Vector[[1.0], Float32[2.834947], [0.0]]
     Vector[[1.0], Float32[3.656587], [0.0]]
     Vector[[1.0], Float32[2.5985842], [0.0]]
     ⋮
     Vector[[1.0], Float32[4.067538], [0.0]]
     Vector[[1.0], Float32[3.02231], [0.0]]
     Vector[[1.0], Float32[2.748292], [0.0]]
     Vector[[1.0], Float32[2.9483426], [0.0]]
     Vector[[1.0], Float32[3.066149], [0.0]]
     Vector[[1.0], Float32[3.6018147], [0.0]]
     Vector[[1.0], Float32[3.0138078], [0.0]]
     Vector[[1.0], Float32[3.5724509], [0.0]]
     Vector[[1.0], Float32[3.117551], [0.0]]
     Vector[[1.0], Float32[2.9670508], [0.0]]
     Vector[[1.0], Float32[3.4107168], [0.0]]
     Vector[[1.0], Float32[3.0252533], [0.0]]

Benchmarks can also be run with parallelization by specifying `parallelizer` argument:

``` julia
# Models:
bmk = benchmark(counterfactual_data; parallelizer = parallelizer)
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
parallelizer = MPIParallelizer(MPI.COMM_WORLD; threaded=true)
```

    Precompiling MPIExt
      ✓ TaijaParallel → MPIExt
      1 dependency successfully precompiled in 4 seconds. 255 already precompiled.
    [ Info: Precompiling MPIExt [48137b38-b316-530b-be8a-261f41e68c23]
    ┌ Warning: Module TaijaParallel with build ID ffffffff-ffff-ffff-0000-b4913f271dd8 is missing from the cache.
    │ This may mean TaijaParallel [bf1c2c22-5e42-4e78-8b6b-92e6c673eeb0] does not support precompilation but is imported by a module that does.
    └ @ Base loading.jl:1948
    [ Info: Skipping precompilation since __precompile__(false). Importing MPIExt [48137b38-b316-530b-be8a-261f41e68c23].
    [ Info: Using `MPI.jl` for multi-processing.

    Running on 1 processes.

    MPIExt.MPIParallelizer(MPI.Comm(1140850688), 0, 1, nothing, true)

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

    Generating counterfactuals ...   8%|▋      |  ETA: 0:00:01 ( 1.19 ms/it)Generating counterfactuals ...  17%|█▏     |  ETA: 0:00:01 ( 1.19 ms/it)Generating counterfactuals ...  26%|█▉     |  ETA: 0:00:01 ( 1.15 ms/it)Generating counterfactuals ...  36%|██▌    |  ETA: 0:00:01 ( 1.15 ms/it)Generating counterfactuals ...  45%|███▏   |  ETA: 0:00:01 ( 1.14 ms/it)Generating counterfactuals ...  55%|███▉   |  ETA: 0:00:01 ( 1.12 ms/it)Generating counterfactuals ...  64%|████▌  |  ETA: 0:00:00 ( 1.12 ms/it)Generating counterfactuals ...  74%|█████▏ |  ETA: 0:00:00 ( 1.11 ms/it)Generating counterfactuals ...  84%|█████▉ |  ETA: 0:00:00 ( 1.11 ms/it)Generating counterfactuals ...  94%|██████▌|  ETA: 0:00:00 ( 1.10 ms/it)Generating counterfactuals ... 100%|███████| Time: 0:00:01 ( 1.10 ms/it)

    1000-element Vector{AbstractCounterfactualExplanation}:
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     ⋮
     CounterfactualExplanation
    Convergence: ✅ after 9 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 6 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 8 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.
     CounterfactualExplanation
    Convergence: ✅ after 7 steps.

To evaluate counterfactuals in parallel, we again use the `parallelize` function:

``` julia
@with_parallelizer parallelizer evaluate(ces)
```

    1000-element Vector{Any}:
     Vector[[1.0], Float32[3.0941274], [0.0]]
     Vector[[1.0], Float32[3.0894346], [0.0]]
     Vector[[1.0], Float32[3.5737448], [0.0]]
     Vector[[1.0], Float32[2.6201036], [0.0]]
     Vector[[1.0], Float32[2.8519764], [0.0]]
     Vector[[1.0], Float32[3.7762523], [0.0]]
     Vector[[1.0], Float32[3.4162796], [0.0]]
     Vector[[1.0], Float32[3.6095932], [0.0]]
     Vector[[1.0], Float32[3.1347957], [0.0]]
     Vector[[1.0], Float32[3.0313473], [0.0]]
     Vector[[1.0], Float32[2.7612567], [0.0]]
     Vector[[1.0], Float32[3.6191392], [0.0]]
     Vector[[1.0], Float32[2.610616], [0.0]]
     ⋮
     Vector[[1.0], Float32[4.0844703], [0.0]]
     Vector[[1.0], Float32[3.0119], [0.0]]
     Vector[[1.0], Float32[2.4461186], [0.0]]
     Vector[[1.0], Float32[3.071967], [0.0]]
     Vector[[1.0], Float32[3.132917], [0.0]]
     Vector[[1.0], Float32[3.5403214], [0.0]]
     Vector[[1.0], Float32[3.0588162], [0.0]]
     Vector[[1.0], Float32[3.5600657], [0.0]]
     Vector[[1.0], Float32[3.2205954], [0.0]]
     Vector[[1.0], Float32[2.896302], [0.0]]
     Vector[[1.0], Float32[3.2603998], [0.0]]
     Vector[[1.0], Float32[3.1369917], [0.0]]

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
using CounterfactualExplanations.Evaluation: benchmark
using CounterfactualExplanations.Models
import MPI

MPI.Init()

data = TaijaData.load_linearly_separable()
counterfactual_data = DataPreprocessing.CounterfactualData(data...)
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
