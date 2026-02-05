# Here we expose functions that are exposed in package extensions, but not in the main package.
# USE VERY SPARINGLY!!! This is a hacky workaround to allow for the use of functions defined in package extensions. If there too much stuff to be exposed, you probably want to work with a separate package instead.
# See here for an example use case: https://discourse.julialang.org/t/how-to-use-functions-defined-in-package-extensions-but-not-the-main-package/99979
# See also this related discussion: https://discourse.julialang.org/t/should-we-define-new-functions-structs-in-an-extension/103361

using CounterfactualExplanations.Models

include("DecisionTreeExt.jl")
include("LaplaceReduxExt.jl")
include("NeuroTreeExt.jl")
include("JEMExt.jl")
