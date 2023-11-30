counterfactual_data = Data.load_overlapping()
bmk = Evaluation.benchmark(counterfactual_data; convergence=:generator_conditions)
@test typeof(bmk()) <: DataFrame
