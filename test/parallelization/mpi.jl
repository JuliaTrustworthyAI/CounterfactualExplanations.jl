using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Evaluation: benchmark
using CounterfactualExplanations.Parallelization
using Logging
using Test

# Initialize MPI
import MPI
MPI.Init()

counterfactual_data = load_linearly_separable()
parallelizer = MPIParallelizer(MPI.COMM_WORLD)
with_logger(NullLogger()) do
    bmk = benchmark(counterfactual_data; parallelizer=parallelizer)
end
MPI.Finalize()
@test MPI.Finalized()

