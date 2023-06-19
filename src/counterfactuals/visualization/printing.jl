function Base.show(io::IO, z::CounterfactualExplanation)
    println(io, "")
    if z.search[:iteration_count] > 0
        if isnothing(z.convergence[:decision_threshold])
            p_path = target_probs_path(z)
            n_reached = findall([
                all(p .>= z.convergence[:decision_threshold]) for p in p_path
            ])
            if length(n_reached) > 0
                printstyled(
                    io,
                    "Threshold reached: $(all(threshold_reached(z)) ? "✅"  : "❌")";
                    bold=true,
                )
                print(" after $(first(n_reached)) steps.\n")
            end
            printstyled(io, "Convergence: $(converged(z) ? "✅"  : "❌")"; bold=true)
            print(" after $(total_steps(z)) steps.\n")
        else
            printstyled(io, "Convergence: $(converged(z) ? "✅"  : "❌")"; bold=true)
            print(" after $(total_steps(z)) steps.\n")
        end
    end
end
