using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Evaluation: benchmark
using CounterfactualExplanations.Parallelization
using Test

# Initialize MPI
import MPI
MPI.Init()

counterfactual_data = load_linearly_separable()
parallelizer = MPIParallelizer(MPI.COMM_WORLD)
bmk = benchmark(counterfactual_data; parallelizer=parallelizer)
MPI.Finalize()
@test MPI.Finalized()

