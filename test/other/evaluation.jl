counterfactual_data = Data.load_overlapping()
bmk = Evaluation.benchmark(counterfactual_data; converge_when=:generator_conditions)
@test typeof(bmk()) <: DataFrame
