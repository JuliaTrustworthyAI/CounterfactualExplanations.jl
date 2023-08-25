function Base.show(io::IO, z::CounterfactualExplanation)
    
    if z.search[:iteration_count] > 0
        println(io, typeof(z))
        printstyled(
            io,
            "Convergence: $(converged(z) ? "✅"  : "❌") after $(total_steps(z)) steps.";
            bold=true,
        )
    else
        println(io, typeof(z))
        printstyled(io, "No search completed."; bold=true)
    end

end
