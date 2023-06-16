ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

include("vision.jl")
@testset "Construction" begin
    for (name, loader) in merge(values(data_catalogue)...)
        @testset "$name" begin
            @test typeof(loader()) == CounterfactualData
        end
    end
end
@testset "Data loading tests" begin
    @testset "load_tabular_data tests" begin
        # Test loading all tabular datasets
        data = Data.load_tabular_data()
        @test typeof(data) == Dict{Symbol,CounterfactualData}
        @test all(k in keys(data) for k in keys(Data.data_catalogue[:tabular]))

        # Test dropping a specific dataset
        data = Data.load_tabular_data(; drop=:california_housing)
        @test !haskey(data, :california_housing)

        # Test loading with specified number of samples
        data = Data.load_tabular_data(1000)
        for dataset in values(data)
            @test size(dataset.X)[2] == 1000  # replace ... with actual size of your data
        end
    end

    @testset "load_synthetic_data tests" begin
        # Test dropping a specific dataset
        data = Data.load_synthetic_data(; drop=:linearly_separable)
        @test !haskey(data, :linearly_separable)

        # Test loading with specified number of samples
        data = Data.load_synthetic_data(500)
        for dataset in values(data)
            @test size(dataset.X)[2] == 500  # replace ... with actual size of your data
        end
    end

    @testset "German credit statlog dataset" begin
        # Test loading german_credit dataset with default parameters
        counterfactual_data = load_german_credit()
        @test size(counterfactual_data.X)[2] == 1000
        @test size(counterfactual_data.X)[1] == 20
        @test size(counterfactual_data.y)[2] == 1000

        # Test loading german_credit dataset with subsampled data
        counterfactual_data = load_german_credit(500)
        @test size(counterfactual_data.X)[2] == 500
        @test size(counterfactual_data.X)[1] == 20
        @test size(counterfactual_data.y)[2] == 500

        # Test case: Load data with n > 1000, expecting an error
        @test_throws ArgumentError load_german_credit(1500)

        # Test case: Load data with n < 1, expecting an error
        @test_throws ArgumentError load_german_credit(0)
        @test_throws ArgumentError load_german_credit(-100)
    end

    @testset "load_california_housing tests" begin
        @test_throws ArgumentError load_california_housing(-1)  # n must be a positive integer or Nothing

        @testset "Check output types and sizes" begin
            result = load_california_housing(100)

            @test isa(result, CounterfactualData)  # the output should be a CounterfactualData object

            @test eltype(result.X) == Float32  # the data should be of type Float32

            @test size(result.X, 2) == 100  # there should be 100 observations

            # Check that the dimensions of X and y match:
            @test size(result.X, 2) == size(result.y, 2)
        end

        @testset "Check data consistency" begin
            # example row:
            # MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,target
            # 3.4698,31.0,5.390243902439025,1.1986062717770034,956.0,3.3310104529616726,33.9,-118.35,1.0
            # we use a hardcoded value for the expected dimensions because the # dataset is very stable and this will greatly speed
            # things up. However, it is still something to be aware of.
            X_dim_expected = 8  # we expect one less column in X, as the target column is not included
            result = load_california_housing(100)

            # Check that the dimension of X is correct:
            @test size(result.X, 1) == X_dim_expected
        end
    end
end
