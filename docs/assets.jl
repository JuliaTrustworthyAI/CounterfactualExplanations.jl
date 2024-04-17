# Since Documenter.jl cannot handle links to anything outside the docs/src directory, we need to copy the LICENSE and CHANGELOG.md files to the src/assets directory: https://discourse.julialang.org/t/make-documenter-jl-understand-where-to-search-local-markdown-links/84012/6?u=pat-alt
symlink(joinpath(@__DIR__, "..", "LICENSE"), "docs/src/LICENSE")
symlink(joinpath(@__DIR__, "..", "CHANGELOG.md"), "docs/src/CHANGELOG.md")
