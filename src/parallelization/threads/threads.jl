"The `ThreadsParallelizer` type is used to parallelize the evaluation of a function using `Threads.@threads`."
struct ThreadsParallelizer <: CounterfactualExplanations.AbstractParallelizer end

include("generate_counterfactual.jl")
include("evaluate.jl")
