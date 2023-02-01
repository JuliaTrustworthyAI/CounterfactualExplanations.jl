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
            "Whiste-Stop Tour" => "tutorials/whistle_stop.md",
            "Handling Data" => "tutorials/data_preprocessing.md",
            "Data Catalogue" => "tutorials/data_catalogue.md",
            "Handling Models" => "tutorials/models.md",
            "Model Catalogue" => "tutorials/model_catalogue.md",
        ],
        "🤓 Explanation" => [
            "Overview" => "explanation/_index.md",
            "Package Architecture" => "explanation/architecture.md",
            "Generators" => [
                "Overview" => "explanation/generators/overview.md", 
                "Generic" => "explanation/generators/generic.md", 
                "Gravitational" => "explanation/generators/gravitational.md", 
                "REVISE" => "explanation/generators/revise.md",
                "DiCE" => "explanation/generators/dice.md",
                "ClaPROAR" => "explanation/generators/clap_roar.md", 
                "Greedy" => "explanation/generators/greedy.md", 
            ],
            "Categorical Features" => "explanation/categorical.md",
            # "Loss functions" => "explanation/loss.md",
        ],
        "🫡 How-To ..." => [
            "Overview" => "how_to_guides/_index.md",
            "... add custom generators" => "how_to_guides/custom_generators.md",
            "... add custom models" => "how_to_guides/custom_models.md",
            # "... explain R/Python models" => "how_to_guides/interop.md",
        ],
        "🧐 Reference" => "_reference.md",
        "🛠 Contribute" => "_contribute.md",
        "📚 Additional Resources" => "assets/_resources.md",
    ]
)

deploydocs(; repo = "github.com/pat-alt/CounterfactualExplanations.jl.git")
