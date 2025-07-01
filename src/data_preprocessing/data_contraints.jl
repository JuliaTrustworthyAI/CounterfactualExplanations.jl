"""
    mutability_constraints(counterfactual_data::CounterfactualData)

A convenience function that returns the mutability constraints. If none were specified, it is assumed that all features are mutable in `:both` directions.
"""
function mutability_constraints(counterfactual_data::CounterfactualData)
    return mutability_constraints(counterfactual_data, counterfactual_data.mutability)
end

"""
    mutability_constraints(counterfactual_data::CounterfactualData, mutability::Nothing)

If nothing is supplied, all features are assumed to be mutable in both directions.
"""
function mutability_constraints(counterfactual_data::CounterfactualData, mutability::Nothing)
    return [:both for i in 1:size(counterfactual_data.X)[1]]
end

"""
mutability_constraints(counterfactual_data::CounterfactualData, mutability::Vector{Symbol})

If `mutability` is already a vector of symbols, it is returned as is. 
"""
function mutability_constraints(counterfactual_data::CounterfactualData, mutability::Vector{Symbol})
    return mutability
end

"""
mutability_constraints(counterfactual_data::CounterfactualData, mutability::Vector{Symbol})

If `mutability` is already a vector of integers, these are assumed to be indices of immutable features. All other features are assumed to be mutable in both directions. This offers a convenient way to quickly specify immutable features.
"""
function mutability_constraints(counterfactual_data::CounterfactualData, mutability::Vector{Int})
    mutability_final = mutability_constraints(counterfactual_data, nothing) # starting point
    for i in mutability
        # Set all referenced indices to immutable:
        mutability_final[i] = :none
    end
    return mutability_final
end


"""
    mutability_constraints(counterfactual_data::CounterfactualData, mutability::Pair{K,V}...) where {K<:Int,V<:Symbol}

If pairs of feature indices (`::Int`) and constraints (`::Symbol`) are supplied, the given constraints are applied to the corresponding features. All other features are assumed to be mutable in both directions.
"""
function mutability_constraints(counterfactual_data::CounterfactualData, mutability::Pair{K,V}...) where {K<:Int,V<:Symbol}
    mutability_final = mutability_constraints(counterfactual_data, nothing) # starting point
    for (k,v) in mutability
        mutability_final[k] = v
    end
    return mutability_final
end

mutability_constraints(counterfactual_data::CounterfactualData, mutability::Tuple{Vararg{Pair{Int, Symbol}}}) = mutability_constraints(counterfactual_data, mutability...)


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
