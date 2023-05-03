using CounterfactualExplanations.Evaluation
using DataFrames

bmk = benchmark(counterfactual_data; converge_when=:generator_conditions)
@test typeof(bmk()) <: DataFrame
