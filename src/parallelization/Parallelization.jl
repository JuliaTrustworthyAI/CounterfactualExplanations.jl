module Parallelization

export MPIParallelizer, @with_parallelizer

import ..CounterfactualExplanations
using CounterfactualExplanations: generate_counterfactual
using CounterfactualExplanations.Evaluation: evaluate

include("utils.jl")

"""
    @with_parallelizer(parallelizer, expr)

This macro can be used to parallelize a function call or block of code. The macro will check that the function is parallelizable and then call `parallelize` with the supplied `parallelizer` and `expr`.
"""
macro with_parallelizer(parallelizer, expr)
    @assert expr.head ∈ (:block, :call) "Expected a block or function call."
    if expr.head == :block
        expr = expr.args[end]
    end

    # Unpack arguments:
    pllr = esc(parallelizer)
    f = esc(expr.args[1])
    args = expr.args[2:end]

    # Split args into positional and keyword arguments:
    aargs = []
    aakws = Pair{Symbol,Any}[]
    for el in args
        if Meta.isexpr(el, :parameters)
            for kw in el.args
                push!(aakws, Pair(kw.args...))
            end
        else
            push!(aargs, el)
        end
    end

    # Escape arguments:
    escaped_args = Expr(:tuple, esc.(aargs)...)

    # Parallelize:
    output = quote
        @assert CounterfactualExplanations.parallelizable($f) "`f` is not a parallelizable process."
        output = CounterfactualExplanations.parallelize(
            $pllr, $f, $escaped_args...; $aakws...
        )
        output
    end
    return output
end

include("mpi.jl")

end