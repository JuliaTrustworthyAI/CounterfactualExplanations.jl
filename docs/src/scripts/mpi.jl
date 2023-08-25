using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Models
using CounterfactualExplanations.Parallelization
import MPI

counterfactual_data = load_linearly_separable()
M = fit_model(counterfactual_data, :Linear)
factual = 1
target = 2
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual), 100)
xs = select_factual(counterfactual_data, chosen)
generator = GenericGenerator()

parallelizer = MPIParallelizer()

output = parallelize(
    parallelizer, 
    generate_counterfactual, 
    xs, target, counterfactual_data, M, generator
)

MPI.Barrier(parallelizer.comm)

MPI.Finalize()