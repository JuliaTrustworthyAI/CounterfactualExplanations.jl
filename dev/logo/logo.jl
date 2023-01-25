using Pkg;
Pkg.activate("dev");

using Colors
using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Models
using Flux
using Luxor
using MLJBase
using StatsBase: sample
using Random

julia_colors = Dict(
    :blue => Luxor.julia_blue,
    :red => Luxor.julia_red,
    :green => Luxor.julia_green,
    :purple => Luxor.julia_purple,
)

function get_data(N=500; seed=1234, centers=2, center_box=(-2 => 2), cluster_std=0.3)

    counterfactual_data = load_blobs(N; seed=seed, centers=centers, center_box=center_box, cluster_std=cluster_std)
    M = fit_model(counterfactual_data, :Linear)

    return counterfactual_data, M
end

function logo_picture(;
    ndots=100,
    frame_size=500,
    ms=frame_size // 50,
    mcolor=(:red, :green),
    margin=-0.1,
    db_color=julia_colors[:purple],
    db_stroke_size=frame_size // 50,
    ce_color=julia_colors[:purple],
    m_alpha=0.2,
    seed=2022,
    cluster_std=0.3,
    γ=0.9,
    η=0.02
)

    # Setup
    n_mcolor = length(mcolor)
    mcolor = getindex.(Ref(julia_colors), mcolor)
    Random.seed!(seed)

    # Data
    counterfactual_data, M = get_data(seed=seed, cluster_std=cluster_std)
    X = counterfactual_data.X
    x = X[1, :]
    y = X[2, :]
    factual = select_factual(counterfactual_data, rand(1:size(counterfactual_data.X, 2)))
    factual_label = predict_label(M, counterfactual_data, factual)[1]
    target = ifelse(factual_label == 1.0, 2.0, 1.0) # opposite label as target

    # Counterfactual:
    generator = GravitationalGenerator(
        opt=Flux.Descent(η),
        decision_threshold=γ
    )
    ce = generate_counterfactual(factual, target, counterfactual_data, M, generator)

    # Dots:
    idx = sample(1:size(X, 2), ndots, replace=false)
    xplot, yplot = (x[idx], y[idx])
    _scale = (frame_size / (2 * maximum(abs.(counterfactual_data.X)))) * (1 - margin)

    # Decision Boundary:
    setline(db_stroke_size)
    sethue(db_color)
    w = collect(Flux.params(M.model))[1]
    a = -w[1] / w[2]
    _bias = collect(Flux.params(M.model))[2][1]
    b = -_bias / w[2]
    _intercept = _scale .* (0, b)
    rule(Point(_intercept[1], _intercept[2]), atan(a))

    # Data
    data_plot = zip(xplot, yplot)
    for i = 1:length(data_plot)
        _point = collect(data_plot)[i]
        _lab = predict_label(M, counterfactual_data, [_point[1], _point[2]])
        color_idx = get_target_index(counterfactual_data.y_levels, _lab[1])
        _x, _y = _scale .* _point
        setcolor(sethue(mcolor[color_idx]...)..., m_alpha)
        circle(Point(_x, _y), ms, action=:fill)
    end

    # Counterfactual path:
    ce_path = CounterfactualExplanations.path(ce)
    lab_path = CounterfactualExplanations.counterfactual_label_path(ce)
    lab_path = Int.(categorical(vec(reduce(vcat, lab_path)[:, :, 1])).refs)
    ce_color = mcolor[lab_path[1]]
    for i in eachindex(ce_path)
        _point = (ce_path[i][1, :, 1][1], ce_path[i][2, :, 1][1])
        color_idx = lab_path[i]
        _x, _y = _scale .* _point
        _alpha = m_alpha + ((1 - m_alpha) * (i / length(ce_path)))
        if i < length(ce_path)
            setcolor(sethue(ce_color)..., _alpha)
        else
            setcolor(sethue(mcolor[color_idx]...)..., _alpha)
        end
        _ms = ms + 1.5 * (ms * (i / length(ce_path)))
        circle(Point(_x, _y), _ms, action=:fill)
    end

end

function draw_small_logo(filename="docs/src/assets/logo.svg", width=500; kwrgs...)
    frame_size = width
    Drawing(frame_size, frame_size, filename)
    background("white")
    origin()
    logo_picture(; kwrgs...)
    finish()
    preview()
end

function draw_wide_logo(
    filename="docs/src/assets/wide_logo.png";
    _pkg_name="Counterfactual Explanations",
    font_size=150,
    font_family="Tamil MN",
    font_fill="transparent",
    font_color=Luxor.julia_blue,
    bg_color="transparent",
    picture_kwargs...
)

    # Setup:
    height = Int(round(font_size * 2.4))
    fontsize(font_size)
    fontface(font_family)
    strs = split(_pkg_name)
    text_col_width = Int(round(maximum(map(str -> textextents(str)[3], strs)) * 1.05))
    width = Int(round(height + text_col_width))
    cw = [height, text_col_width]
    cells = Luxor.Table(height, cw)
    ms = Int(round(height / 10))
    db_stroke_size = Int(round(height / 50))

    Drawing(width, height, filename)
    origin()
    background(bg_color)

    # Picture:
    @layer begin
        translate(cells[1])
        logo_picture(
            ;
            frame_size=height,
            margin=0.1,
            ms=ms,
            db_stroke_size=db_stroke_size,
            picture_kwargs...,
        )
    end

    # Text:
    @layer begin
        translate(cells[2])
        fontsize(font_size)
        fontface(font_family)
        tiles = Tiler(cells.colwidths[2], height, length(strs), 1)
        for (pos, n) in tiles
            @layer begin
                translate(pos)
                setline(Int(round(db_stroke_size / 5)))
                sethue(font_fill)
                textoutlines(strs[n], O, :path, valign=:middle, halign=:center)
                sethue(font_color)
                strokepath()
            end
        end
    end

    finish()
    preview()
end

picture_kwargs = (
    seed=961,
    margin=-0.3,
    ndots=250,
    ms=12,
    cluster_std=0.3,
    η=0.02,
    γ=0.95
)

draw_small_logo(; picture_kwargs...)
draw_wide_logo(; picture_kwargs...)
