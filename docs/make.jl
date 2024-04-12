using CounterfactualExplanations
using Documenter

include("setup_docs.jl")

DocMeta.setdocmeta!(
    CounterfactualExplanations, :DocTestSetup, :(setup_docs); recursive=true
)

makedocs(;
    modules=[CounterfactualExplanations],
    authors="Patrick Altmeyer",
    repo="https://github.com/juliatrustworthyai/CounterfactualExplanations.jl/blob/{commit}{path}#{line}",
    sitename="CounterfactualExplanations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliatrustworthyai.github.io/CounterfactualExplanations.jl",
        assets=String[],
        size_threshold_ignore=["reference.md"],
    ),
    pages=[
        "🏠 Home" => "index.md",
        "🫣 Tutorials" => [
            "Overview" => "tutorials/index.md",
            "Simple Example" => "tutorials/simple_example.md",
            "Whiste-Stop Tour" => "tutorials/whistle_stop.md",
            "Handling Data" => "tutorials/data_preprocessing.md",
            "Data Catalogue" => "tutorials/data_catalogue.md",
            "Handling Models" => "tutorials/models.md",
            "Model Catalogue" => "tutorials/model_catalogue.md",
            "Handling Generators" => "tutorials/generators.md",
            "Evaluating Explanations" => "tutorials/evaluation.md",
            "Benchmarking Explanations" => "tutorials/benchmarking.md",
            "Parallelization" => "tutorials/parallelization.md",
        ],
        "🤓 Explanation" => [
            "Overview" => "explanation/index.md",
            "Package Architecture" => "explanation/architecture.md",
            "Generators" => [
                "Overview" => "explanation/generators/overview.md",
                "Generic" => "explanation/generators/generic.md",
                "ClaPROAR" => "explanation/generators/clap_roar.md",
                "CLUE" => "explanation/generators/clue.md",
                "DiCE" => "explanation/generators/dice.md",
                "FeatureTweak" => "explanation/generators/feature_tweak.md",
                "Gravitational" => "explanation/generators/gravitational.md",
                "Greedy" => "explanation/generators/greedy.md",
                "GrowingSpheres" => "explanation/generators/growing_spheres.md",
                "PROBE" => "explanation/generators/probe.md",
                "REVISE" => "explanation/generators/revise.md",
            ],
            "Optimisers" => [
                "Overview" => "explanation/optimisers/overview.md",
                "JSMA" => "explanation/optimisers/jsma.md",
            ],
            "Categorical Features" => "explanation/categorical.md",
            # "Loss functions" => "explanation/loss.md",
        ],
        "🫡 How-To ..." => [
            "Overview" => "how_to_guides/index.md",
            "... add custom generators" => "how_to_guides/custom_generators.md",
            "... add custom models" => "how_to_guides/custom_models.md",
            # "... explain R/Python models" => "how_to_guides/interop.md",
        ],
        "⛓️ Extensions" => ["Overview" => "extensions/index.md"],
        "🧐 Reference" => "reference.md",
        "🛠 Contribute" => "contribute.md",
        "📚 Additional Resources" => "assets/resources.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl", devbranch="main"
)
