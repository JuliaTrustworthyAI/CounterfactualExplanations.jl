include("generative_models.jl")
include("pretrained.jl")
include("flux/mlp.jl")

# Extensions:
if VERSION >= v"1.9"
    # Extension currently only works with Julia 1.9 and above (https://github.com/Evovest/NeuroTreeModels.jl/pull/7)
    include("neurotree/neurotree.jl")
end
if VERSION >= v"1.7"
    # Extension currently only works with Julia 1.7.
    include("laplace_redux/laplace_redux.jl")
end
include("decision_tree/decision_tree.jl")

include("utils.jl")

@testset "Standard models for synthetic data" begin
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            X = value[:data].X
            for (likelihood, model) in value[:models]
                name = string(likelihood)
                @testset "$name" begin
                    @testset "Verify correctness of likelihood field" begin
                        @test model[:model].likelihood == value[:data].likelihood
                    end
                    @testset "Matrix of inputs" begin
                        @test size(Models.logits(model[:model], X))[2] == size(X, 2)
                        @test size(Models.probs(model[:model], X))[2] == size(X, 2)
                    end
                    @testset "Vector of inputs" begin
                        @test size(Models.logits(model[:model], X[:, 1]), 2) == 1
                        @test size(Models.probs(model[:model], X[:, 1]), 2) == 1
                    end
                end
            end
        end
    end
end

@testset "Non-standard models for synthetic data" begin
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            X = value[:data].X

            # Test the DecisionTreeModel model
            model = Models.fit_model(value[:data], :DecisionTreeModel)
            name = "DecisionTreeModel"
            @testset "$name" begin
                @testset "Verify correctness of likelihood field" begin
                    @test model.likelihood == value[:data].likelihood
                end
                @testset "Matrix of inputs" begin
                    @test size(Models.logits(model, X))[2] == size(X, 2)
                    @test size(Models.probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(Models.logits(model, X[:, 1]), 2) == 1
                    @test size(Models.probs(model, X[:, 1]), 2) == 1
                end
            end

            # Test the RandomForest model
            model = Models.fit_model(value[:data], :RandomForestModel)
            name = "RandomForestModel"
            @testset "$name" begin
                @testset "Verify correctness of likelihood field" begin
                    @test model.likelihood == value[:data].likelihood
                end
                @testset "Matrix of inputs" begin
                    @test size(Models.logits(model, X))[2] == size(X, 2)
                    @test size(Models.probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(Models.logits(model, X[:, 1]), 2) == 1
                    @test size(Models.probs(model, X[:, 1]), 2) == 1
                end
            end

            # Test the LaplaceReduxModel
            model = Models.fit_model(value[:data], CounterfactualExplanations.LaplaceReduxModel())

            @testset "Verify correctness of likelihood field for LaplaceRedux" begin
                @test model.likelihood == :classification_multi
            end
        end
    end
end

@testset "Test for errors" begin
    @test_throws AssertionError Models.Model(MLP(); likelihood=:regression)

    # test Flux models
    @test_throws ArgumentError Models.FluxModel("dummy"; likelihood=:regression)
    @test_throws ArgumentError Models.FluxEnsemble("dummy"; likelihood=:regression)

    data = TaijaData.load_linearly_separable()
    counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
        data[1], data[2]
    )
    X, y = DataPreprocessing.preprocess_data_for_mlj(counterfactual_data)
end
