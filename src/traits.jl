"A base type for a style of process."
abstract type ProcessStyle end

"By default all types of have this trait."
struct NotParallel <: ProcessStyle end
ProcessStyle(::Type) = NotParallel()

"Processes that can be parallelized have this trait."
struct IsParallel <: ProcessStyle end

# Implementing trait behaviour:
parallelizable(x::T) where {T} = parallelizable(ProcessStyle(T), x)
parallelizable(::IsParallel, x) = true
parallelizable(::NotParallel, x) = false

macro with_parallelizer(parallelizer::Nothing, f::Function, args...)
    return esc(f(args...))
end

"""
    parallelize(
        parallelizer::nothing,
        f::Function,
        args...;
        kwargs...,
    )

If no `AbstractParallelizer` has been supplied, just call the function. 
"""
parallelize(parallelizer::Nothing, f::Function, args...; kwargs...) = f(args...; kwargs...)