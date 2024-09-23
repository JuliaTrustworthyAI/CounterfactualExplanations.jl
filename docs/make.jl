using CounterfactualExplanations
using Changelog
using Documenter

include("assets.jl")

# Generate a Documenter-friendly changelog from CHANGELOG.md
Changelog.generate(
    Changelog.Documenter(),
    joinpath(@__DIR__, "..", "CHANGELOG.md"),
    joinpath(@__DIR__, "src", "release-notes.md");
    repo="juliatrustworthyai/CounterfactualExplanations.jl",
)

makedocs(;
    doctest=false,
    modules=[CounterfactualExplanations],
    authors="Patrick Altmeyer",
    repo=Documenter.Remotes.GitHub("juliatrustworthyai", "CounterfactualExplanations.jl"),
    sitename="CounterfactualExplanations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliatrustworthyai.github.io/CounterfactualExplanations.jl/stable",
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
            "Convergence" => "tutorials/convergence.md",
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
                "MINT" => "explanation/generators/mint.md",
                "T-CREx" => "explanation/generators/tcrex.md",
            ],
            "Evaluation" => [
                "Overview" => "explanation/evaluation/overview.md",
                "Plausibility and Faithfulness" => "explanation/evaluation/faithfulness.md",
            ],
            "Optimisers" => [
                "Overview" => "explanation/optimisers/overview.md",
                "JSMA" => "explanation/optimisers/jsma.md",
            ],
            "Categorical Features" => "explanation/categorical.md",
        ],
        "🫡 How-To ..." => [
            "Overview" => "how_to_guides/index.md",
            "... add custom generators" => "how_to_guides/custom_generators.md",
            "... add custom models" => "how_to_guides/custom_models.md",
        ],
        "⛓️ Extensions" => [
            "Overview" => "extensions/index.md",
            "NeuroTrees" => "extensions/neurotree.md",
            "LaplaceRedux" => "extensions/laplace_redux.md",
        ],
        "🧐 Reference" => "reference.md",
        "🛠 Contribute" => "contribute.md",
        "📚 Additional Resources" => "assets/resources.md",
        "Release Notes" => "release-notes.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl.git", devbranch="main"
)
