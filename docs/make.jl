using CounterfactualExplanations
using Documenter

DocMeta.setdocmeta!(CounterfactualExplanations, :DocTestSetup, :(using CounterfactualExplanations); recursive=true)

makedocs(;
    modules=[CounterfactualExplanations],
    authors="Patrick Altmeyer",
    repo="https://github.com/pat-alt/CounterfactualExplanations.jl/blob/{commit}{path}#{line}",
    sitename="CounterfactualExplanations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pat-alt.github.io/CounterfactualExplanations.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Motivating example" => "cats_dogs.md",
        "Tutorials" =>
            [
                "Binary target" => "tutorials/binary.md",
                "Models" => "tutorials/models.md",
                "Multi-class target" => "tutorials/multi.md",
                "Loss functions" => "tutorials/loss.md"
            ],
        "More examples" =>
            [
                "Image data" => [
                    "MNIST" => "examples/image/MNIST.md"
                ],
                # "Time series" => [
                #     "UCR ECG200" => "examples/timeseries/UCR_ECG200.md"
                # ]
            ],
        "Reference" => "reference.md"
    ],
)

deploydocs(;
    repo="github.com/pat-alt/CounterfactualExplanations.jl",
    devbranch="main"
)
