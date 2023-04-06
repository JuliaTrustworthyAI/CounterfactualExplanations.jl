using CounterfactualExplanations.Evaluation
using DataFrames

bmk = benchmark(counterfactual_data; converge_when=:generator_conditions)
@test typeof(bmk.ces) <: Vector{CounterfactualExplanation}
@test typeof(bmk()) <: DataFrame
