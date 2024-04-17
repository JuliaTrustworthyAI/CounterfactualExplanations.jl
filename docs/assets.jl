# Since Documenter.jl cannot handle links to anything outside the docs/src directory, we need to copy the LICENSE and CHANGELOG.md files to the src/assets directory: https://discourse.julialang.org/t/make-documenter-jl-understand-where-to-search-local-markdown-links/84012/6?u=pat-alt
cp(joinpath(@__DIR__, "..", "LICENSE"), joinpath(@__DIR__, "src/LICENSE"); force=true)
cp(
    joinpath(@__DIR__, "..", "CHANGELOG.md"),
    joinpath(@__DIR__, "src/CHANGELOG.md");
    force=true,
)
