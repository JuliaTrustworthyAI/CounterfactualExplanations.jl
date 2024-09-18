"T-CREx counterfactual generator class."
mutable struct TCRExGenerator <: AbstractNonGradientBasedGenerator 
    ρ::AbstractFloat
    τ::AbstractFloat
    forest::Bool
end

function TCRExGenerator(ρ::AbstractFloat=0.2, τ::AbstractFloat=0.9; forest::Bool=false)
    return TCRExGenerator(ρ, τ, forest)
end

function grow_surrogate end

function extract_rules end

const DOC_TCREx = "For details see Bewley et al. ([2024](https://arxiv.org/abs/2405.18875))."

"""
    rule_feasibility(rule, X)

Computes the feasibility of a rule ``R_i`` for a given dataset. Feasibility is defined as fraction of the data points that satisfy the rule. $DOC_TCREx
"""
function rule_feasibility(rule, X)
    checks = 0
    for x in eachcol(X)
        per_feature = [lb <= x[i] < ub for (i, (lb,ub)) in enumerate(rule)]
        checks += Int(all(per_feature))
    end
    return checks / size(X, 2)
end

"""
    rule_feasibility(rule, ce::CounterfactualExplanation)

Overloads the `rule_feasibility` function for [`CounterfactualExplanation`](@ref) objects.
"""
rule_feasibility(rule, ce::CounterfactualExplanation) = rule_feasibility(rule, ce.data.X)

"""
    rule_accuracy(rule, X, fx, target)

Computes the accuracy of the rule on the data `X` for predicted outputs `fx` and the `target`. Accuracy is defined as the fraction of points contained by the rule, for which predicted values match the target. $DOC_TCREx
"""
function rule_accuracy(rule, X, fx, target)
    ingroup = 0
    checks = 0
    for (j, x) in enumerate(eachcol(X))
        # Check that sample is contained in the rule:
        per_feature = [lb <= x[i] < ub for (i, (lb, ub)) in enumerate(rule)]
        if all(per_feature)
            # Add to group:
            ingroup += 1
            # Check that 𝒻 gives an output in target:
            checks += Int(fx[j] ∈ target)
        end
    end
    return checks / ingroup
end

"""
    issubrule(rule, otherrule)

Checks if the `rule` hyperrectangle is a subset of the `otherrule` hyperrectangle. $DOC_TCREx
"""
function issubrule(rule, otherrule)
    return all([y[1] <= x[1] && x[2] <= y[2] for (x, y) in zip(rule, otherrule)])
end

function max_valid(rules, X, fx, target, τ)

    # Consider only rules that meet accuracy threshold:
    candidate_rules = findall(rule_accuracy.(rules, (X,), (fx,), (target,)) .>= τ) |>
        idx -> rules[idx]

    max_valid_rules = []

    for (i,rule) in enumerate(candidate_rules)
        other_rules = candidate_rules[setdiff(eachindex(candidate_rules),i)]
        if all([!issubrule(rule, otherrule) for otherrule in other_rules])
            push!(max_valid_rules, rule)
        end
    end

    return max_valid_rules
end