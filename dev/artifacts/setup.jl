setup = quote

    using Pkg; Pkg.activate("dev/artifacts")

    # Deps:
    using CounterfactualExplanations
    using CSV

    # Utils
    include("dev/artifacts/utils.jl")

end