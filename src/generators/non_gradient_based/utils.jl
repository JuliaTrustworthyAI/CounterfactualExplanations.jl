function conditions_satisfied(
    generator::GrowingSpheresGenerator, ce::AbstractCounterfactualExplanation
)   
    return generator.flag == :converged
end