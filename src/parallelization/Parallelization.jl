module Parallelization

using ..CounterfactualExplanations

const CanBeParallelised = Union{
    typeof(CounterfactualExplanations.generate_counterfactual),
    typeof(CounterfactualExplanations.Evaluation.evaluate),
}

include("mpi.jl")

end