using CounterfactualExplanations.Data
using CounterfactualExplanations.Models
using Printf

"""
    _load_synthetic()

Loads synthetic data, models, and generators.
"""
function _load_synthetic()
    # Data:
    data_sets = Dict(
        :classification_binary => load_linearly_separable(),
        :classification_multi => load_multi_class(),
    )
    # Models
    synthetic = Dict()
    for (likelihood, data) in data_sets
        models = Dict()
        for (model_name, model) in standard_models_catalogue
            M = fit_model(data, model_name)
            models[model_name] = Dict(:raw_model => M.model, :model => M)
        end
        synthetic[likelihood] = Dict(:models => models, :data => data)
    end
    return synthetic
end

function get_target(counterfactual_data::CounterfactualData, factual_label::RawTargetType)
    target = rand(
        counterfactual_data.y_levels[counterfactual_data.y_levels .!= factual_label]
    )
    return target
end

"""
    create_new_pytorch_model(data::CounterfactualData, model_path::String)

Creates a new PyTorch model and saves it to a Python file.
"""
function create_new_pytorch_model(data::CounterfactualData, model_path::String)
    in_size = size(data.X)[1]
    out_size = size(data.y)[1]

    class_str = """
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear($(in_size), 32),
                nn.Sigmoid(),
                nn.Linear(32, $(out_size))
            )

        def forward(self, x):
            return self.model(x)
    """

    open(model_path, "w") do f
        @printf(f, "%s", class_str)
    end

    return nothing
end

"""
    train_and_save_pytorch_model(data::CounterfactualData, model_location::String, pickle_path::String)

Trains a PyTorch model and saves it to a pickle file.
"""
function train_and_save_pytorch_model(
    data::CounterfactualData, model_location::String, pickle_path::String
)
    sys = PythonCall.pyimport("sys")

    if !in(model_location, sys.path)
        sys.path.append(model_location)
    end

    importlib = PythonCall.pyimport("importlib")
    neural_network_class = importlib.import_module("neural_network_class")
    importlib.reload(neural_network_class)
    NeuralNetwork = neural_network_class.NeuralNetwork
    model = NeuralNetwork()

    x_python, y_python = CounterfactualExplanations.DataPreprocessing.preprocess_python_data(
        data
    )

    optimizer = torch.optim.Adam(model.parameters(); lr=0.1)
    loss_fun = torch.nn.BCEWithLogitsLoss()

    # Training
    for _ in 1:100
        # Compute prediction and loss:
        output = model(x_python).squeeze()
        loss = loss_fun(output, y_python.t())
        println("training...")
        # Backpropagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end

    torch.save(model, pickle_path)
    return nothing
end

function remove_file(file_path::String)
    try
        rm(file_path)  # removes the file
        println("File $file_path removed successfully.")
        return nothing
    catch e
        throw(ArgumentError("Error occurred while removing file $file_path: $e"))
    end
end
