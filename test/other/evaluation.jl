using TaijaData: load_moons, load_circles
using CounterfactualExplanations.Evaluation:
    Benchmark, evaluate, validity, distance_measures
using CounterfactualExplanations.Objectives: distance

# Dataset
data = TaijaData.load_overlapping()
counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
    data[1], data[2]
)

# Factual and target:
n_individuals = 5
M = fit_model(counterfactual_data, :Linear)
target = 2
factual = 1
ids = rand(findall(predict_label(M, counterfactual_data) .== factual), n_individuals)
xs = select_factual(counterfactual_data, ids)
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)
generator = GenericGenerator()
ces = generate_counterfactual(
    xs, target, counterfactual_data, M, generator; num_counterfactuals=5
)
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
# Meta data:
meta_data = Dict(:generator => "Generic", :model => "MLP")
meta_data = [meta_data for i in 1:length(ces)]
# Pre-trained models:
models = Dict(
    :MLP => fit_model(counterfactual_data, :MLP),
    :Linear => fit_model(counterfactual_data, :Linear),
)
# Generators:
generators = Dict(
    :Generic => GenericGenerator(),
    :Gravitational => GravitationalGenerator(),
    :Wachter => WachterGenerator(),
    :ClaPROAR => ClaPROARGenerator(),
)

@testset "Evaluation" begin
    @test typeof(evaluate(ce; measure=validity)) <: Vector
    @test typeof(evaluate(ce; measure=distance)) <: Vector
    @test typeof(evaluate(ce; measure=distance_measures)) <: Vector
    @test typeof(evaluate(ce)) <: Vector
    @test typeof(evaluate.(ces)) <: Vector
    @test typeof(evaluate.(ces; report_each=true)) <: Vector
    @test typeof(evaluate.(ces; output_format=:Dict, report_each=true)) <: Vector{<:Dict}
    @test typeof(evaluate.(ces; output_format=:DataFrame, report_each=true)) <:
        Vector{<:DataFrame}

    # Faithfulness and plausibility:
    faith = Evaluation.faithfulness(ce)
    faith = Evaluation.faithfulness(ce; choose_lowest_energy=true)
    faith = Evaluation.faithfulness(ce; choose_random=true)
    faith = Evaluation.faithfulness(ce; cosine=true)
    delete!(ce.search, :energy_sampler)
    delete!(ce.M.fitresult.other, :energy_sampler)
    faith = Evaluation.faithfulness(ce; nwarmup=100)
    plaus = Evaluation.plausibility(ce)
    plaus = Evaluation.plausibility(ce; choose_random=true)
    @test true
end

@testset "Benchmarking" begin
    bmk = Evaluation.benchmark(counterfactual_data; convergence=:generator_conditions)

    @testset "Basics" begin
        @test typeof(bmk()) <: DataFrame
        @test typeof(bmk(; agg=nothing)) <: DataFrame
        @test typeof(vcat(bmk, bmk)) <: Benchmark
    end

    @testset "Different methods" begin
        @test typeof(benchmark(ces)) <: Benchmark
        @test typeof(benchmark(ces; meta_data=meta_data)) <: Benchmark
        @test typeof(
            benchmark(x, target, counterfactual_data; models=models, generators=generators)
        ) <: Benchmark
    end

    @testset "Full one" begin
        # Data:
        datasets = Dict(
            :moons => CounterfactualData(load_moons()...),
            :circles => CounterfactualData(load_circles()...),
        )

        # Models:
        models = Dict(:MLP => MLP, :Linear => Linear)

        # Generators:
        generators = Dict(:Generic => GenericGenerator(), :Greedy => GreedyGenerator())

        using CounterfactualExplanations.Evaluation: distance_measures
        bmks = []
        for (dataname, dataset) in datasets
            bmk = benchmark(
                dataset; models=models, generators=generators, measure=distance_measures
            )
            push!(bmks, bmk)
        end
        bmk = vcat(bmks[1], bmks[2]; ids=collect(keys(datasets)))
        @test typeof(bmk) <: Benchmark
    end
end

@testset "Serialization" begin
    global_serializer(Serializer())
    @test _serialization_state == true
    global_serializer(NullSerializer())
    @test _serialization_state == false
end
