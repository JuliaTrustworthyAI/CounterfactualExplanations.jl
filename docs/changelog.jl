using Changelog

Changelog.generate(
    Changelog.CommonMark(),
    joinpath(@__DIR__, "..", "CHANGELOG.md");
    repo="juliatrustworthyai/CounterfactualExplanations.jl",
)
