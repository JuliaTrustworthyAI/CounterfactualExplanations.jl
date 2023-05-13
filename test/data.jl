ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "Construction" begin
    for (name, loader) in merge(values(data_catalogue)...)
        @testset "$name" begin
            @test typeof(loader()) == CounterfactualData
        end
    end
end

@testset "Vision tests" begin
    # Test loading CIFAR10 dataset with default parameters
    counterfactual_data = load_cifar_10()
    @test size(counterfactual_data.features)[1] == 50000
    @test size(counterfactual_data.features)[2] == 3072
    @test size(counterfactual_data.targets)[1] == 50000
    @test counterfactual_data.domain == (-1.0, 1.0)
    @test counterfactual_data.standardize == false
    @test eltype(counterfactual_data.features) == Float32
    @test eltype(counterfactual_data.targets) == Float32

    # Test loading CIFAR10 dataset with subsampled data
    counterfactual_data = load_cifar_10(1000)
    @test size(counterfactual_data.features)[1] == 1000
    @test size(counterfactual_data.features)[2] == 3072
    @test size(counterfactual_data.targets)[1] == 1000
    @test counterfactual_data.domain == (-1.0, 1.0)
    @test counterfactual_data.standardize == false
    @test eltype(counterfactual_data.features) == Float32
    @test eltype(counterfactual_data.targets) == Float32

    # Test loading CIFAR10 test dataset
    counterfactual_data = load_cifar_10_test()
    @test size(counterfactual_data.features)[1] == 10000
    @test size(counterfactual_data.features)[2] == 3072
    @test size(counterfactual_data.targets)[1] == 10000
    @test counterfactual_data.domain == (-1.0, 1.0)
    @test counterfactual_data.standardize == false
    @test eltype(counterfactual_data.features) == Float32
    @test eltype(counterfactual_data.targets) == Float32
end