using Pkg;
Pkg.activate("dev");

using Colors
using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Models
using Distributions
using EasyFit
using Flux
using LinearAlgebra
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
bg_color = RGBA(julia_colors[:blue]..., 0.25)

function (fit::EasyFit.Cubic)(x::Real)
    a = fit.a
    b = fit.b
    c = fit.c
    d = fit.d
    return a * x^3 + b * x^2 + c * x + d
end

function get_data(
    N=600;
    seed=2023,
    cluster_std=0.3,
    generator=GenericGenerator(),
    model=:Linear,
    n_steps=nothing
)


    Random.seed!(seed)
    
    n_per = Int.(round(N / 3))
    d1 = MvNormal([0, 1], UniformScaling(cluster_std)(2))
    x1 = rand(d1, n_per)
    d2 = MvNormal([-1, -1], UniformScaling(cluster_std)(2))
    x2 = rand(d2, n_per)
    d3 = MvNormal([1, -1], UniformScaling(cluster_std)(2))
    x3 = rand(d3, n_per)
    X = hcat(x1, x2, x3)
    y = categorical(reduce(vcat, permutedims(repeat([1, 2, 3], 1, n_per))))
    counterfactual_data = CounterfactualData(X, y)
    M = fit_model(counterfactual_data, model)
    X = counterfactual_data.X
    x = X[1, :]
    y = X[2, :]
    factual = select_factual(counterfactual_data, rand(1:size(counterfactual_data.X, 2)))
    factual_label = predict_label(M, counterfactual_data, factual)[1]
    target = counterfactual_data.y_levels[counterfactual_data.y_levels.!=factual_label][1]

    # Decision boundary:
    n_range = 100
    xlims, ylims = extrema(X, dims=2)
    _zoom = 0.5 * maximum(abs.(X))
    xrange = range(xlims[1] - _zoom, xlims[2] + _zoom, length=n_range)
    yrange = range(ylims[1] - _zoom, ylims[2] + _zoom, length=n_range)
    target_idx = get_target_index(counterfactual_data.y_levels, target)
    yhat = [MLJBase.predict(M, counterfactual_data, [x, y])[target_idx] for x in xrange, y in yrange]
    db_points = findall(0.0 .< yhat .- 0.5 .< 0.1)
    db_points = map(p -> [xrange[p[1]], yrange[p[2]]], db_points)
    _X = reduce(hcat, db_points)
    x1 = _X[1, :]
    x2 = _X[2, :]
    fit = fitcubic(x1, x2)
    _yhat = fit.(xrange)
    db_points = map(x -> [x[1], x[2]], zip(collect(xrange), _yhat))


    # Counterfactual:
    T = isnothing(n_steps) ? 100 : n_steps
    ce = generate_counterfactual(factual, target, counterfactual_data, M, generator; T=T)

    return x, y, M, ce, db_points
end

function logo_picture(;
    ndots=100,
    frame_size=500,
    ms=frame_size // 50,
    mcolor=(:red, :purple, :green),
    margin=-0.1,
    db_color=RGBA(julia_colors[:blue]..., 0.5),
    db_stroke_size=frame_size // 75,
    switch_ce_color=true,
    m_alpha=0.2,
    seed=2022,
    cluster_std=0.3,
    generator=GenericGenerator(),
    model=:MLP,
    bg=true,
    bg_color="transparent",
    clip_border=true,
    border_color=julia_colors[:blue],
    border_stroke_size=db_stroke_size,
    n_steps=nothing,
    smooth=4
)

    # Setup
    n_mcolor = length(mcolor)
    mcolor = getindex.(Ref(julia_colors), mcolor)
    Random.seed!(seed)

    # Background 
    if bg
        circle(O, frame_size // 2, :clip)
        setcolor(bg_color)
        box(Point(0, 0), frame_size, frame_size, action=:fill)
        setcolor(border_color..., 1.0)
        circle(O, frame_size // 2, :stroke)
    end

    # Data
    x, y, M, ce, db_points = get_data(
        seed=seed,
        cluster_std=cluster_std,
        generator=generator, model=model, n_steps=n_steps
    )

    # Dots:
    idx = sample(1:length(x), ndots, replace=false)
    xplot, yplot = (x[idx], y[idx])
    _scale = (frame_size / (2 * maximum(abs.(ce.data.X)))) * (1 - margin)

    # Decision Boundary:
    setline(db_stroke_size)
    setcolor(db_color)
    order_db = sortperm([x[1] for x in db_points])
    pgon = [Point((_scale .* (_point[1], _point[2]))...) for _point in db_points[order_db]]
    np = makebezierpath(pgon, smoothing=smooth)
    drawbezierpath(np, action=:stroke, close=false)

    # Data
    data_plot = zip(xplot, yplot)
    for i = 1:length(data_plot)
        _point = collect(data_plot)[i]
        _lab = predict_label(M, ce.data, [_point[1], _point[2]])
        color_idx = get_target_index(ce.data.y_levels, _lab[1])
        _x, _y = _scale .* _point
        setcolor(sethue(mcolor[color_idx]...)..., m_alpha)
        circle(Point(_x, _y), ms, action=:fill)
    end

    # Counterfactual path:
    ce_path = CounterfactualExplanations.path(ce)
    for i in eachindex(ce_path)
        _point = (ce_path[i][1, :, 1][1], ce_path[i][2, :, 1][1])
        _lab = predict_label(M, ce.data, [_point[1], _point[2]])
        color_idx = get_target_index(ce.data.y_levels, _lab[1])
        _x, _y = _scale .* _point
        _alpha = i != length(ce_path) ? m_alpha : 1.0
        _ms = i != length(ce_path) ? db_stroke_size : 1.5 * ms 
        setcolor(sethue(mcolor[color_idx]...)..., _alpha)
        circle(Point(_x, _y), _ms, action=:fill)
    end

end

function draw_small_logo(filename="docs/src/assets/logo.svg", width=500; bg_color="transparent", kwrgs...)
    frame_size = width
    Drawing(frame_size, frame_size, filename)
    if !isnothing(bg_color)
        background(bg_color)
    end
    origin()
    logo_picture(; kwrgs...)
    finish()
    preview()
end

function animate_small_logo(filename="docs/src/assets/logo.gif", width=500; bg_color="transparent", kwrgs...)
    frame_size = width
    anim = Movie(frame_size, frame_size, "logo", 1:10)
    function backdrop(scene, framenumber)
        background(bg_color)
    end
    function frame(scene, framenumber)
        logo_picture(; kwrgs..., m_alpha=1.0, switch_ce_color=false, bg_color=julia_colors[:blue], n_steps=framenumber)
    end
    animate(
        anim, 
        [
            Scene(anim, backdrop, 1:10),
            Scene(anim, frame, 1:10, easingfunction=easeinoutcubic)
        ],
        creategif=true,
        pathname=filename
    )
end

function draw_wide_logo(
    filename="docs/src/assets/wide_logo.png";
    _pkg_name="Counterfactual Explanations",
    font_size=150,
    font_family="Tamil MN",
    font_fill=bg_color,
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
        Luxor.translate(cells[1])
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
        Luxor.translate(cells[2])
        fontsize(font_size)
        fontface(font_family)
        tiles = Tiler(cells.colwidths[2], height, length(strs), 1)
        for (pos, n) in tiles
            @layer begin
                Luxor.translate(pos)
                setline(Int(round(db_stroke_size / 5)))
                setcolor(font_fill)
                textoutlines(strs[n], O, :path, valign=:middle, halign=:center)
                fillpreserve()
                setcolor(font_color...,1.0)
                strokepath()
            end
        end
    end

    finish()
    preview()
end

_seed = rand(1:1000)
picture_kwargs = (
    seed=_seed,
    margin=0.1,
    ndots=30,
    ms=30,
    cluster_std=0.05,
    clip_border=true,
    m_alpha=0.5,
    generator=GravitationalGenerator(
        decision_threshold=0.75,
        opt=Descent(0.01)
    )
)

draw_small_logo(; picture_kwargs...)
draw_wide_logo(; picture_kwargs...)
