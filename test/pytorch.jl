# Using PyTorch models is supported only for Julia versions >= 1.8
if VERSION >= v"1.8"
    torch = PythonCall.pyimport("torch")
    @testset "PyTorch model test" begin
        model_file = "neural_network_class"
        class_name = "NeuralNetwork"
        model_location = "$(pwd())"
        model_path = "$(pwd())/neural_network_class.py"
        pickle_path = "$(pwd())/pretrained_model.pt"

        for (key, value) in synthetic
            name = string(key)
            @testset "$name" begin
                data = value[:data]
                X = data.X

                # Create and save model in the model_path directory
                create_new_model(data, model_path)
                train_and_save_model(data, model_location, pickle_path)
                model_loaded = CounterfactualExplanations.Models.pytorch_model_loader(
                    model_location, model_file, class_name, pickle_path
                )

                model_pytorch = CounterfactualExplanations.Models.PyTorchModel(
                    model_loaded, data.likelihood
                )

                @testset "$name" begin
                    @testset "Matrix of inputs" begin
                        @test size(logits(model_pytorch, X))[2] == size(X, 2)
                        @test size(probs(model_pytorch, X))[2] == size(X, 2)
                    end
                    @testset "Vector of inputs" begin
                        @test size(logits(model_pytorch, X[:, 1]), 2) == 1
                        @test size(probs(model_pytorch, X[:, 1]), 2) == 1
                    end
                end
                remove_file(model_path)
                remove_file(pickle_path)
            end
        end
    end
end