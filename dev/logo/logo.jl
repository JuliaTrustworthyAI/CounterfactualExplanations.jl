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

const julia_colors = Dict(
    :blue => Luxor.julia_blue,
    :red => Luxor.julia_red,
    :green => Luxor.julia_green,
    :purple => Luxor.julia_purple,
)

function get_data(N=500; center_box=(-5 => 5), cluster_std=0.5)

    counterfactual_data = load_linearly_separable(N)
    M = fit_model(counterfactual_data, :Linear)

    return counterfactual_data, M
end

function logo_picture(;
    ndots=3,
    frame_size=500,
    ms=frame_size // 10,
    mcolor=(:red, :green, :purple),
    margin=0.1,
    fun=f(x) = x * cos(x),
    xmax=2.5,
    noise=0.5,
    ged_data=get_data,
    ntrue=50,
    gt_color=julia_colors[:blue],
    gt_stroke_size=5,
    interval_color=julia_colors[:blue],
    interval_alpha=0.2,
    seed=2022
)

    # Setup
    n_mcolor = length(mcolor)
    mcolor = getindex.(Ref(julia_colors), mcolor)
    Random.seed!(seed)

    # Data
    counterfactual_data, M = get_data()
    factual = select_factual(counterfactual_data, rand(1:size(counterfactual_data.X,2)))
    factual_label = predict_label(M, counterfactual_data, factual)[1]
    target = ifelse(factual_label == 1.0, 0.0, 1.0) # opposite label as target

    # Counterfactual:
    generator = GenericGenerator()
    ce = generate_counterfactual(factual, target, counterfactual_data, M, generator)

    _scale = (frame_size / (2 * maximum(x))) * (1 - margin)

    # Decision Boundary:
    setline(gt_stroke_size)
    sethue(gt_color)
    w = collect(Flux.params(M.model))[1]
    b = collect(Flux.params(M.model))[2][1]
    a = -w[2] / w[1]
    _ymin = _scale .* (b + a * (-xmax))
    _ymax = _scale .* (b + a * (xmax))
    pmin = Point(_scale * (-xmax), _ymin)
    pmax = Point(_scale * xmax, _ymax)
    line(pmin, pmax, action=:stroke)

    # Data
    # data_plot = zip(xplot, yplot)
    # for i = 1:length(data_plot)
    #     _x, _y = _scale .* collect(data_plot)[i]
    #     color_idx = i % n_mcolor == 0 ? n_mcolor : i % n_mcolor
    #     sethue(mcolor[color_idx]...)
    #     circle(Point(_x, _y), ms, action=:fill)
    # end

end

function draw_small_logo(filename="docs/src/assets/logo.svg"; width=500)
    frame_size = width
    Drawing(frame_size, frame_size, filename)
    origin()
    logo_picture(frame_size=frame_size)
    finish()
    preview()
end

function draw_wide_logo_new(
    filename="docs/src/assets/wide_logo.png";
    _pkg_name="Conformal Prediction",
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
    gt_stroke_size = Int(round(height / 50))

    Drawing(width, height, filename)
    origin()
    background(bg_color)

    # Picture:
    @layer begin
        translate(cells[1])
        logo_picture(
            frame_size=height,
            margin=0.1,
            ms=ms,
            gt_stroke_size=gt_stroke_size,
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
                setline(Int(round(gt_stroke_size / 5)))
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

draw_wide_logo_new()
