bmk = Evaluation.benchmark(counterfactual_data; converge_when=:generator_conditions)
@test typeof(bmk()) <: DataFrame
