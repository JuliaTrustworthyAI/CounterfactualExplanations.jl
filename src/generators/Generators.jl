# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..Models
using ..Losses
using Flux
using LinearAlgebra

export Generator, GenericGenerator, GreedyGenerator, update_recourse, convergence

include("functions.jl")

end