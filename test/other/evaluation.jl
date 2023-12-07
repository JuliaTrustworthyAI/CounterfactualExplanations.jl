data = TaijaData.load_overlapping()
counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
    data[1], data[2]
)
bmk = Evaluation.benchmark(counterfactual_data; converge_when=:generator_conditions)
@test typeof(bmk()) <: DataFrame
