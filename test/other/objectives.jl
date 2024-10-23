using CounterfactualExplanations
using CounterfactualExplanations.Models: fit_model
using CounterfactualExplanations.Objectives
using DecisionTree

# Counteractual data and model:
counterfactual_data = CounterfactualData(TaijaData.load_linearly_separable()...)
M = fit_model(counterfactual_data, :Linear)
target = 2
factual = 1
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)

# Search:
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

@testset "Losses" begin
    for (lname, lfun) in Objectives.losses_catalogue
        @test lfun(ce) isa AbstractFloat
    end
end

@testset "Penalties" begin
    for (pname, pfun) in Objectives.penalties_catalogue
        @test pfun(ce) isa AbstractFloat
    end

    @testset "EnergyDifferential" begin
        M_tree = fit_model(counterfactual_data, :DecisionTreeModel)
        target = 2
        factual = 1
        chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
        x = select_factual(counterfactual_data, chosen)

        # Search:
        generator = FeatureTweakGenerator()
        ce = generate_counterfactual(x, target, counterfactual_data, M_tree, generator)

        # EnergyDifferential:
        @test_throws NotImplementedModel EnergyDifferential()(ce)
    end
end
