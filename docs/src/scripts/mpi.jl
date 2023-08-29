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