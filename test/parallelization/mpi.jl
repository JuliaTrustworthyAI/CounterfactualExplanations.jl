using TaijaData
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation: benchmark
using CounterfactualExplanations.Parallelization
using Logging
using Test

# Initialize MPI
using MPI: MPI
MPI.Init()

data = TaijaData.load_linearly_separable()
counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
    data[1], data[2]
)
parallelizer = MPIParallelizer(MPI.COMM_WORLD)
with_logger(NullLogger()) do
    bmk = benchmark(counterfactual_data; parallelizer=parallelizer)
end
MPI.Finalize()
@test MPI.Finalized()
