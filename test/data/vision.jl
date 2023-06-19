@testset "Vision tests" begin
    # Test loading CIFAR10 dataset with default parameters
    @testset "cifar 10" begin
        counterfactual_data = load_cifar_10()
        @test size(counterfactual_data.X)[2] == 50000
        @test size(counterfactual_data.X)[1] == 3072
        @test size(counterfactual_data.y)[2] == 50000
        @test all(
            counterfactual_data.domain[i] == (-1.0, 1.0) for
            i in eachindex(counterfactual_data.domain)
        )
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32

        # Test loading CIFAR10 dataset with subsampled data
        counterfactual_data = load_cifar_10(1000)
        @test size(counterfactual_data.X)[2] == 1000
        @test size(counterfactual_data.X)[1] == 3072
        @test size(counterfactual_data.y)[2] == 1000
        @test all(
            counterfactual_data.domain[i] == (-1.0, 1.0) for
            i in eachindex(counterfactual_data.domain)
        )
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32

        # Test loading CIFAR10 test dataset
        counterfactual_data = load_cifar_10_test()
        @test size(counterfactual_data.X)[2] == 10000
        @test size(counterfactual_data.X)[1] == 3072
        @test size(counterfactual_data.y)[2] == 10000
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32
    end

    @testset "fashion mnist" begin
        counterfactual_data = load_fashion_mnist()
        @test size(counterfactual_data.X)[2] == 60000
        @test size(counterfactual_data.X)[1] == 784
        @test size(counterfactual_data.y)[2] == 60000
        @test all(
            counterfactual_data.domain[i] == (-1.0, 1.0) for
            i in eachindex(counterfactual_data.domain)
        )
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32

        # Test loading Fashion MNIST dataset with subsampled data
        counterfactual_data = load_fashion_mnist(1000)
        @test size(counterfactual_data.X)[2] == 1000
        @test size(counterfactual_data.X)[1] == 784
        @test size(counterfactual_data.y)[2] == 1000
        @test all(
            counterfactual_data.domain[i] == (-1.0, 1.0) for
            i in eachindex(counterfactual_data.domain)
        )
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32

        # Test loading Fashion MNIST test dataset
        counterfactual_data = load_fashion_mnist_test()
        @test size(counterfactual_data.X)[2] == 10000
        @test size(counterfactual_data.X)[1] == 784
        @test size(counterfactual_data.y)[2] == 10000
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32
    end

    @testset "mnist" begin
        counterfactual_data = load_mnist()
        @test size(counterfactual_data.X)[2] == 60000
        @test size(counterfactual_data.X)[1] == 784
        @test size(counterfactual_data.y)[2] == 60000
        @test all(
            counterfactual_data.domain[i] == (-1.0, 1.0) for
            i in eachindex(counterfactual_data.domain)
        )
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32

        # Test loading MNIST dataset with subsampled data
        counterfactual_data = load_mnist(1000)
        @test size(counterfactual_data.X)[2] == 1000
        @test size(counterfactual_data.X)[1] == 784
        @test size(counterfactual_data.y)[2] == 1000
        @test all(
            counterfactual_data.domain[i] == (-1.0, 1.0) for
            i in eachindex(counterfactual_data.domain)
        )
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32

        # Test loading MNIST test dataset
        counterfactual_data = load_mnist_test()
        @test size(counterfactual_data.X)[2] == 10000
        @test size(counterfactual_data.X)[1] == 784
        @test size(counterfactual_data.y)[2] == 10000
        @test counterfactual_data.standardize == false
        @test eltype(counterfactual_data.X) == Float32
    end
end
