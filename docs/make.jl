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
                "Overview" => "tutorials/index.md",
                "Binary target" => "tutorials/binary.md",
                "Custom models" => "tutorials/models.md",
                "Multi-class target" => "tutorials/multi.md",
                "Custom generators" => "tutorials/generators.md",
                "Mutability constraints" => "tutorials/mutability.md",
                "Interoperability" => "tutorials/interop.md"
            ],
        "Counterfactual Generators" =>
            [
                "Latent Space Search" => "generators/gradient_based/latent_space_generator.md"
            ],
        "More examples" =>
            [
                "Image data" => [
                    "MNIST" => "examples/image/MNIST.md"
                ],
            ],
        "Contributor's Guide" =>
            [
                "Overview" => "contributing/index.md",
                "Interoperability" => "contributing/interop.md",
                "Loss functions" => "contributing/loss.md"
            ],
        "Reference" => "reference.md",
        "Additional Resources" => "resources/resources.md"
    ],
)

deploydocs(;
    repo="github.com/pat-alt/CounterfactualExplanations.jl.git"
)
