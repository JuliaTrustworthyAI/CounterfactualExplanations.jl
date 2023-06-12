"""
    mutability_constraints(counterfactual_data::CounterfactualData)

A convenience function that returns the mutability constraints. If none were specified, it is assumed that all features are mutable in `:both` directions.
"""
function mutability_constraints(counterfactual_data::CounterfactualData)
    return if isnothing(counterfactual_data.mutability)
        [:both for i in 1:size(counterfactual_data.X)[1]]
    else
        counterfactual_data.mutability
    end
end

"""
    apply_domain_constraints(counterfactual_data::CounterfactualData, x::AbstractArray) 

A subroutine that is used to apply the predetermined domain constraints.
"""
function apply_domain_constraints(counterfactual_data::CounterfactualData, x::AbstractArray)

    # Continuous variables:
    if !isnothing(counterfactual_data.domain)
        for i in counterfactual_data.features_continuous
            x[i] = clamp(
                x[i], counterfactual_data.domain[i][1], counterfactual_data.domain[i][2]
            )
        end
    end

    return x
end
