using MPI
MPI.Init()

using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Evaluation: benchmark
using CounterfactualExplanations.Models
using CounterfactualExplanations.Parallelization
using Logging
using Serialization

if MPI.Comm_rank(MPI.COMM_WORLD) != 0
    global_logger(NullLogger())
else
    @info "Disabling logging on non-root processes."
end

counterfactual_data = load_linearly_separable()
M = fit_model(counterfactual_data, :Linear)
factual = 1
target = 2
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual), 100)
xs = select_factual(counterfactual_data, chosen)
generator = GenericGenerator()

parallelizer = MPIParallelizer(MPI.COMM_WORLD; threaded=true)

bmk = with_logger(NullLogger()) do
    benchmark(counterfactual_data; parallelizer=parallelizer)
end

n = 250

@info "Benchmarking with MPI"
time_mpi = @elapsed with_logger(NullLogger()) do
    benchmark(counterfactual_data; parallelizer=parallelizer, n_individuals=n)
end

@info "Benchmarking without MPI"
time_wo = @elapsed with_logger(NullLogger()) do
    benchmark(counterfactual_data; parallelizer=nothing, n_individuals=n)
end

Serialization.serialize("docs/src/scripts/mpi_benchmark.jls", bmk)
time_bmk = Dict(
    :time_mpi => time_mpi,
    :time_wo => time_wo,
    :n => n,
    :n_cores => MPI.Comm_size(MPI.COMM_WORLD),
)
Serialization.serialize("docs/src/scripts/mpi_benchmark_times.jls", time_bmk)

MPI.Finalize()
