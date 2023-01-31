using CounterfactualExplanations
using Documenter

DocMeta.setdocmeta!(
    CounterfactualExplanations,
    :DocTestSetup,
    :(using CounterfactualExplanations);
    recursive = true,
)

makedocs(;
    modules=[CounterfactualExplanations],
    authors="Patrick Altmeyer",
    repo="https://github.com/pat-alt/CounterfactualExplanations.jl/blob/{commit}{path}#{line}",
    sitename="CounterfactualExplanations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pat-alt.github.io/CounterfactualExplanations.jl",
        assets=String[]
    ),
    pages=[
        "🏠 Home" => "index.md",
        "🫣 Tutorials" => [
            "Overview" => "tutorials/_index.md",
            "Whiste-Stop Tour" => "tutorials/whistle-stop.md",
            "Datasets" => "tutorials/data.md",
            "Models" => "tutorials/models.md",
            "Data Pre-Processing" => "tutorials/data_preprocessing.md",
        ],
        "🫡 How-To ..." => [
            "Overview" => "how_to_guides/_index.md",
            "... explain an image classifer" => "how_to_guides/mnist.qmd",
            "... add custom models" => "how_to_guides/custom_models.md",
            "... add custom generators" => "how_to_guides/custom_generators.md",
            "... explain R/Python models" => "how_to_guides/interop.md",
        ],
        "🤓 Explanation" => [
            "Overview" => "explanation/_index.md",
            "Package Architecture" => "explanation/architecture.md",
            "Generators" => [
                "REVISE" => "explanation/generators/latent_space_generator.md",
                "DiCE" => "explanation/generators/dice.md",
            ],
            "Loss functions" => "contributing/loss.md",
        ],
        "🧐 Reference" => "_reference.md",
        "🛠 Contribute" => "contribute.md",
        "📚 Additional Resources" => "assets/_resources.md",
    ]
)

deploydocs(; repo = "github.com/pat-alt/CounterfactualExplanations.jl.git")
