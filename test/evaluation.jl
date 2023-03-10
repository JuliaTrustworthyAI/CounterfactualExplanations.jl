using CounterfactualExplanations.Evaluation
using DataFrames

bmk = benchmark(counterfactual_data)
@test typeof(bmk.counterfactual_explanations) <: Vector{CounterfactualExplanation}
@test typeof(bmk()) <: DataFrame