using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Evaluation: benchmark
using CounterfactualExplanations.Parallelization
using Test
counterfactual_data = load_linearly_separable()
parallelizer = MPIParallelizer()
bmk = benchmark(counterfactual_data; parallelizer=parallelizer)
@test true