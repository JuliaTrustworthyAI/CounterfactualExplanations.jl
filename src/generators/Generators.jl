# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..Models
using ..Losses
using Flux
using LinearAlgebra

export Generator, GenericGenerator, GreedyGenerator, update_recourse, condtions_satisified

include("functions.jl")

end