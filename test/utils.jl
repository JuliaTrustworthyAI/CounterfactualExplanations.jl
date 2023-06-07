using CounterfactualExplanations.Data
using CounterfactualExplanations.Models
using Printf

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
        for (model_name, model) in model_catalogue
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

function create_new_model(data::CounterfactualData)
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

    open("$(pwd())/neural_network_class.py", "w") do f
        @printf(f, "%s", class_str)
    end
end

function train_and_save_model(data::CounterfactualData, model_path::String, pickle_path::String)
    sys = PythonCall.pyimport("sys")
    if !in(model_path, sys.path)
        sys.path.append(model_path)
    end

    importlib = PythonCall.pyimport("importlib")
    neural_network_class = importlib.import_module("neural_network_class")
    importlib.reload(neural_network_class)
    NeuralNetwork = neural_network_class.NeuralNetwork
    model = NeuralNetwork()

    x_python, y_python = CounterfactualExplanations.DataPreprocessing.preprocess_python_data(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fun = torch.nn.BCEWithLogitsLoss()

    # Training
    for epoch in 1:100
        # Compute prediction and loss:
        output = model(x_python).squeeze()
        loss = loss_fun(output, y_python.t())
        # Backpropagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end

    torch.save(model, pickle_path)
end

function remove_python_file(filename::String)
    try
        rm(filename)  # removes the file
        println("File $filename removed successfully.")
    catch e
        println("Error occurred while removing file $filename: $e")
    end
end

function remove_pickle_file(filename::String)
    try
        rm(filename)  # removes the file
        println("File $filename removed successfully.")
    catch e
        println("Error occurred while removing file $filename: $e")
    end
end
