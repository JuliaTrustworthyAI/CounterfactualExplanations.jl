using CounterfactualExplanations
using Documenter

include("setup_docs.jl")

@info "Setting up doctest metadata ..."
DocMeta.setdocmeta!(
    CounterfactualExplanations, :DocTestSetup, :($setup_docs); recursive=true
)
doctest(CounterfactualExplanations)
