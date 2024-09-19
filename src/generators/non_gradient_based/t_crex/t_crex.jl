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

"""
    max_valid(rules, X, fx, target, τ)

Returns the maximal-valid rules for a given `target` and accuracy threshold `τ`. $DOC_TCREx
"""
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

"""
    partition_bounds(rules, dim::Int)

Computes the set of (unique) bounds for each rule in `rules` along the `dim`-th dimension. $DOC_TCREx
"""
function partition_bounds(rules, dim::Int)
    lb = [-Inf]
    ub = [Inf]
    for rule in rules
        lb_dim = rule[dim][1]
        ub_dim = rule[dim][2]
        push!(lb, lb_dim)
        push!(ub, ub_dim)
    end
    return lb, ub
end

"""
    induced_grid(rules)

Computes the induced grid of the given rules. $DOC_TCREx.
"""
function induced_grid(rules)
    D = length(rules[1])
    return Base.Iterators.product(
        [Generators.partition_bounds(rules, d) |> x -> zip(x[1], x[2]) for d in 1:D]...
    ) |> unique
end

"""
    rule_contains(rule, X)

Returns the subet of `X` that is contained by rule ``R_i``. $DOC_TCREx
"""
function rule_contains(rule, X)
    contained = Bool[]
    for x in eachcol(X)
        per_feature = [lb <= x[i] < ub for (i, (lb, ub)) in enumerate(rule)]
        push!(contained, Int(all(per_feature)))
    end
    return X[:, contained]
end

@doc raw"""
    prototype(rule, X)

Picks an arbitrary point ``x^C \in X`` (i.e. prototype) from the subet of ``X`` that is contained by rule ``R_i``. $DOC_TCREx
"""
function prototype(rule, X)
    return rule_contains(rule, X) |> X -> X[:,rand(1:size(X,2))]
end

"""
    rule_changes(rule, x)

Computes the number of feature changes necessary for `x` to be contained by rule ``R_i``. $DOC_TCREx
"""
function rule_changes(rule, x)
    return sum([x[i] <= lb || ub < x[i] for (i, (lb, ub)) in enumerate(rule)])
end

"""
    rule_cost(rule, x, X)

Computes the cost for ``x`` to be contained by rule ``R_i``, where cost is defined as `rule_changes(rule, x) - rule_feasibility(rule, X)`. $DOC_TCREx 
"""
function rule_cost(rule, x, X)
    return rule_changes(rule, x) - rule_feasibility(rule, X)
end

"""
    cre(rules, x, X)

Computes the counterfactual rule explanations (CRE) for a given point ``x`` and a set of ``rules``, where the ``rules`` correspond to the set of maximal-valid rules for some given target.  $DOC_TCREx
"""
function cre(rules, x, X; return_index::Bool=false)
    idx = [rule_cost(rule, x, X) for rule in rules] |> argmin
    if return_index
        return idx
    else
        return rules[idx]
    end
end