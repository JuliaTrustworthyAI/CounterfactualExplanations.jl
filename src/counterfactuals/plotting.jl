using MLJBase
using Plots
using Parameters
using SliceMap

function Plots.plot(
    counterfactual_explanation::CounterfactualExplanation;
    alpha_ = 0.5,
    plot_up_to::Union{Nothing,Int} = nothing,
    plot_proba::Bool = false,
    kwargs...,
)

    T = total_steps(counterfactual_explanation)
    T =
        isnothing(plot_up_to) ? total_steps(counterfactual_explanation) :
        minimum([plot_up_to, T])
    T += 1
    ingredients = set_up_plots(
        counterfactual_explanation;
        alpha = alpha_,
        plot_proba = plot_proba,
        kwargs...,
    )

    for t ∈ 1:T
        final_state = t == T
        plot_state(counterfactual_explanation, t, final_state; ingredients...)
    end

    plt =
        plot_proba ? plot(ingredients.p1, ingredients.p2; kwargs...) :
        plot(ingredients.p1; kwargs...)

    return plt

end

"""
    animate_path(counterfactual_explanation::CounterfactualExplanation, path=tempdir(); plot_proba::Bool=false, kwargs...)

Animate the counterfactual path.
"""
function animate_path(
    counterfactual_explanation::CounterfactualExplanation,
    path = tempdir();
    alpha_ = 0.5,
    plot_up_to::Union{Nothing,Int} = nothing,
    plot_proba::Bool = false,
    kwargs...,
)
    T = total_steps(counterfactual_explanation)
    T =
        isnothing(plot_up_to) ? total_steps(counterfactual_explanation) :
        minimum([plot_up_to, T])
    T += 1
    ingredients = set_up_plots(
        counterfactual_explanation;
        alpha = alpha_,
        plot_proba = plot_proba,
        kwargs...,
    )

    anim = @animate for t ∈ 1:T
        final_state = t == T
        plot_state(counterfactual_explanation, t, final_state; ingredients...)
        plot_proba ? plot(ingredients.p1, ingredients.p2; kwargs...) :
        plot(ingredients.p1; kwargs...)
    end
    return anim
end


function plot_state(
    counterfactual_explanation::CounterfactualExplanation,
    t::Int,
    final_sate::Bool;
    kwargs...,
)
    args = PlotIngredients(; kwargs...)
    x1 = vec(mapslices(X -> X[1], args.path_embedded[t], dims = (1, 2)))
    x2 = vec(mapslices(X -> X[2], args.path_embedded[t], dims = (1, 2)))
    y = Int.(vec(selectdim(args.path_labels.refs, 1, t)))
    n_ = counterfactual_explanation.num_counterfactuals
    label_ = reshape(["C$i" for i = 1:n_], 1, n_)
    if !final_sate
        scatter!(args.p1, x1, x2, colour = y; ms = 5, label = "")
    else
        scatter!(args.p1, x1, x2, colour = y; ms = 10, label = "")
        if n_ > 1
            label_1 = vec([text(lab, 5) for lab in label_])
            annotate!(x1, x2, label_1)
        end
    end
    if args.plot_proba
        probs_ = reshape(reduce(vcat, args.path_probs[1:t]), t, n_)
        if t == 1 && n_ > 1
            label_2 = label_
        else
            label_2 = ""
        end
        plot!(
            args.p2,
            probs_,
            label = label_2,
            color = reshape(1:n_, 1, n_),
            title = "p(y=$(counterfactual_explanation.target))",
        )
    end
end

@with_kw struct PlotIngredients
    p1::Any
    p2::Any
    path_embedded::Any
    path_labels::Any
    path_probs::Any
    alpha::Any
    plot_proba::Any
end

function set_up_plots(
    counterfactual_explanation::CounterfactualExplanation;
    alpha,
    plot_proba,
    kwargs...,
)
    p1 = Models.plot(
        counterfactual_explanation.M,
        counterfactual_explanation.data;
        target = counterfactual_explanation.target,
        alpha = alpha,
        kwargs...,
    )
    p2 = plot(xlims = (1, total_steps(counterfactual_explanation) + 1), ylims = (0, 1))
    path_embedded = embed_path(counterfactual_explanation)
    path_labels =
        MLJBase.categorical(reduce(vcat, (counterfactual_label_path(counterfactual_explanation))))
    path_probs = target_probs_path(counterfactual_explanation)
    output = (
        p1 = p1,
        p2 = p2,
        path_embedded = path_embedded,
        path_labels = path_labels,
        path_probs = path_probs,
        alpha = alpha,
        plot_proba = plot_proba,
    )
    return output
end
