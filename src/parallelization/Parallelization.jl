module Parallelization

import ..CounterfactualExplanations

include("utils.jl")

include("mpi.jl")
export MPIParallelizer

end