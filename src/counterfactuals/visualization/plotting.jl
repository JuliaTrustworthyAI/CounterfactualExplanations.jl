"""
    Plots.plot(
        ce::CounterfactualExplanation;
        alpha_ = 0.5,
        plot_up_to::Union{Nothing,Int} = nothing,
        plot_proba::Bool = false,
        kwargs...,
    )

Calling `plot` on an instance of type `CounterfactualExplanation` returns a plot that visualises the entire counterfactual path. For multi-dimensional input data, the data is first compressed into two dimensions. The decision boundary is then approximated using using a Nearest Neighbour classifier. This is still somewhat experimental at the moment.


# Examples

```julia-repl
# Search:
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

plot(ce)
```
"""
function Plots.plot(
    ce::CounterfactualExplanation;
    alpha_=0.5,
    plot_up_to::Union{Nothing,Int}=nothing,
    plot_proba::Bool=false,
    kwargs...,
)
    max_iter = total_steps(ce)
    max_iter = if isnothing(plot_up_to)
        total_steps(ce)
    else
        minimum([plot_up_to, max_iter])
    end
    max_iter += 1
    ingredients = set_up_plots(ce; alpha=alpha_, plot_proba=plot_proba, kwargs...)

    for t in 1:max_iter
        final_state = t == max_iter
        plot_state(ce, t, final_state; ingredients...)
    end

    plt = if plot_proba
        Plots.plot(ingredients.p1, ingredients.p2; kwargs...)
    else
        Plots.plot(ingredients.p1; kwargs...)
    end

    return plt
end

"""
    animate_path(ce::CounterfactualExplanation, path=tempdir(); plot_proba::Bool=false, kwargs...)

Returns and animation of the counterfactual path.

# Examples

```julia-repl
# Search:
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)

animate_path(ce)
```
"""
function animate_path(
    ce::CounterfactualExplanation,
    path=tempdir();
    alpha_=0.5,
    plot_up_to::Union{Nothing,Int}=nothing,
    plot_proba::Bool=false,
    kwargs...,
)
    max_iter = total_steps(ce)
    max_iter = if isnothing(plot_up_to)
        total_steps(ce)
    else
        minimum([plot_up_to, max_iter])
    end
    max_iter += 1
    ingredients = set_up_plots(ce; alpha=alpha_, plot_proba=plot_proba, kwargs...)

    anim = @animate for t in 1:max_iter
        final_state = t == max_iter
        plot_state(ce, t, final_state; ingredients...)
        if plot_proba
            plot(ingredients.p1, ingredients.p2; kwargs...)
        else
            plot(ingredients.p1; kwargs...)
        end
    end
    return anim
end

"""
    plot_state(
        ce::CounterfactualExplanation,
        t::Int,
        final_sate::Bool;
        kwargs...
    )

Helper function that plots a single step of the counterfactual path.
"""
function plot_state(ce::CounterfactualExplanation, t::Int, final_sate::Bool; kwargs...)
    args = PlotIngredients(; kwargs...)
    x1 = selectdim(args.path_embedded[t], 1, 1)
    x2 = selectdim(args.path_embedded[t], 1, 2)
    y = args.path_labels[t]
    _c = levelcode.(y)
    n_ = ce.num_counterfactuals
    label_ = reshape(["C$i" for i in 1:n_], 1, n_)
    if !final_sate
        scatter!(args.p1, x1, x2; group=y, colour=_c, ms=5, label="")
    else
        scatter!(args.p1, x1, x2; group=y, colour=_c, ms=10, label="")
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
            probs_;
            label=label_2,
            color=reshape(1:n_, 1, n_),
            title="p(y=$(ce.target))",
        )
    end
end

"A container used for plotting."
@with_kw struct PlotIngredients
    p1::Any
    p2::Any
    path_embedded::Any
    path_labels::Any
    path_probs::Any
    alpha::Any
    plot_proba::Any
end

"""
    set_up_plots(
        ce::CounterfactualExplanation;
        alpha,
        plot_proba,
        kwargs...
    )

A helper method that prepares data for plotting.
"""
function set_up_plots(ce::CounterfactualExplanation; alpha, plot_proba, kwargs...)
    p1 = Models.plot(ce.M, ce.data; target=ce.target, alpha=alpha, kwargs...)
    p2 = plot(; xlims=(1, total_steps(ce) + 1), ylims=(0, 1))
    path_embedded = embed_path(ce)
    path_labels = counterfactual_label_path(ce)
    y_levels = ce.data.y_levels
    path_labels = map(x -> categorical(x; levels=y_levels), path_labels)
    path_probs = target_probs_path(ce)
    output = (
        p1=p1,
        p2=p2,
        path_embedded=path_embedded,
        path_labels=path_labels,
        path_probs=path_probs,
        alpha=alpha,
        plot_proba=plot_proba,
    )
    return output
end
