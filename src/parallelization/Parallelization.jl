module Parallelization

import ..CounterfactualExplanations
using CounterfactualExplanations: generate_counterfactual
using CounterfactualExplanations.Evaluation: evaluate

include("utils.jl")

include("mpi.jl")
export MPIParallelizer, @with_parallelizer

end