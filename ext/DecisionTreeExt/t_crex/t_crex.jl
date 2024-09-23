using CounterfactualExplanations: CRE, Rule
using CounterfactualExplanations.DataPreprocessing
using CounterfactualExplanations.Models

include("utils.jl")

function (generator::Generators.TCRExGenerator)(
    target::RawTargetType,
    data::DataPreprocessing.CounterfactualData,
    M::Models.AbstractModel
)

    # Setup:    
    X = data.X
    fx = predict_label(M, data)

    # (a) ##############################

    # Surrogate:
    model, fitresult = grow_surrogate(generator, data, M)

    # Extract rules:
    R = extract_rules(fitresult[1]) 

    # (b) ##############################
    R_max = max_valid(R, X, fx, target, generator.τ)

    # (c) ##############################
    _grid = induced_grid(R_max)
    
    # (d) ##############################
    xs = prototype.(_grid, (X,); pick_arbitrary=false)
    Rᶜ = cre.((R_max,), xs, (X,); return_index=true)

    # (e) - (f) ########################
    bounds = partition_bounds(R_max)
    tree = classify_prototypes(hcat(xs...)', Rᶜ, bounds)
    R_final, labels = extract_leaf_rules(tree)

    # Construct CRE:
    output = CRE(
        target,
        data,
        M,
        generator,
        Rule.(R_max),
        Rule.(R_final),
        nothing,
    )

    return output
    
end