

``` @meta
CurrentModule = CounterfactualExplanations 
```

``` julia
include("$(pwd())/docs/setup_docs.jl")
eval(setup_docs)

# Packages
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.Models
using Flux
using JointEnergyModels
using MLJFlux
using TaijaBase.Samplers: PMC, SGLD, ImproperSGLD
using TaijaData
```

# Faithfulness

## Synthetic Data

- ☐ Joint Energy Model

### Joint Energy Model

``` julia
n_obs = 1000
X, y = TaijaData.load_blobs(n_obs; cluster_std=0.1, center_box=(-1. => 1.))
data = CounterfactualData(X, y)

n_hidden = 16
_batch_size = Int(round(n_obs/10))
epochs = 100
M = Models.fit_model(
    data,:JEM;
    builder=MLJFlux.MLP(
        hidden=(n_hidden, n_hidden, n_hidden), 
        σ=Flux.swish
    ),
    batch_size=_batch_size,
    finaliser=Flux.softmax,
    loss=Flux.Losses.crossentropy,
    jem_training_params=(
        α=[1.0,1.0,1e-1],
        verbosity=10,
    ),
    epochs=epochs,
    sampling_steps=30,
)
```

``` julia
# Select a factual instance:
target = 2
factual = 1
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Search parameters:
opt = Adam(0.01)
conv = GeneratorConditionsConvergence()

# Generic Generator:
λ₁ = 0.1
generator = GenericGenerator(opt=opt, λ=λ₁)
ce = generate_counterfactual(x, target, data, M, generator; convergence=conv, num_counterfactuals=5)
faith = Evaluation.faithfulness(ce)
p1 = plot(ce; zoom=-1)
X̂ = ce.search[:energy_sampler][ce.target].posterior
scatter!(X̂[1, :], X̂[2, :]; label="X|y=$target", shape=:star5, ms=10, title="Generic Generator ($(round(faith, digits=5)))", color=3, alpha=0.1)
scatter!(ce.x′[1,:], ce.x′[2,:]; label="Counterfactual", shape=:star1, ms=20, color=4)

# Search:
λ₂ = 0.5
generator = ECCoGenerator(opt=opt; λ=[λ₁, λ₂])
ce = generate_counterfactual(x, target, data, M, generator; convergence=conv, num_counterfactuals=5)
faith = Evaluation.faithfulness(ce)
p2 = plot(ce; zoom=-1)
X̂ = ce.search[:energy_sampler][ce.target].posterior
scatter!(X̂[1, :], X̂[2, :]; label="X|y=$target", shape=:star5, ms=10, title="ECCo Generator ($(round(faith, digits=5)))", color=3, alpha=0.1)
scatter!(ce.x′[1,:], ce.x′[2,:]; label="Counterfactual", shape=:star1, ms=20, color=4)

plot(p1, p2; size=(1000, 400))
```

### Deep Ensemble

``` julia
n_obs = 1000
X, y = TaijaData.load_blobs(n_obs; cluster_std=0.1, center_box=(-1. => 1.))
data = CounterfactualData(X, y)
flux_training_params.n_epochs = 1
M = Models.fit_model(data,:DeepEnsemble)
CounterfactualExplanations.reset!(flux_training_params)
```

``` julia
# Select a factual instance:
target = 2
factual = 1
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Search parameters:
opt = Adam(0.1)
conv = GeneratorConditionsConvergence()

# Generic Generator:
generator = GenericGenerator(opt=opt)
ce = generate_counterfactual(x, target, data, M, generator; convergence=conv, initialization=:identity)
faith = Evaluation.faithfulness(ce)
X̂ = ce.search[:energy_sampler][ce.target].posterior
p1 = plot(ce, zoom=-1)
scatter!(X̂[1, :], X̂[2, :]; label="X|y=$target", shape=:star5, ms=10, title="Generic Generator ($(round(faith, digits=5)))", color=3, alpha=0.1)
scatter!(ce.x′[1,:], ce.x′[2,:]; label="Counterfactual", shape=:star1, ms=20, color=4)
_lim = maximum(abs.(X̂))
xlims, ylims = (-_lim, _lim), (-_lim, _lim)
p3 = plot(ce; xlims=xlims, ylims=ylims)
scatter!(X̂[1, :], X̂[2, :]; label="X|y=$target", shape=:star5, ms=10, title="Generic Generator ($(round(faith, digits=5)))", color=3, alpha=0.1)
scatter!(ce.x′[1,:], ce.x′[2,:]; label="Counterfactual", shape=:star1, ms=20, color=4)

# Search:
generator = ECCoGenerator(opt=opt)
ce = generate_counterfactual(x, target, data, M, generator; convergence=conv, initialization=:identity)
faith = Evaluation.faithfulness(ce)
X̂ = ce.search[:energy_sampler][ce.target].posterior
p2 = plot(ce, zoom=-1)
scatter!(X̂[1, :], X̂[2, :]; label="X|y=$target", shape=:star5, ms=10, title="ECCo Generator ($(round(faith, digits=5)))", color=3, alpha=0.1)
scatter!(ce.x′[1,:], ce.x′[2,:]; label="Counterfactual", shape=:star1, ms=20, color=4)
_lim = maximum(abs.(X̂))
xlims, ylims = (-_lim, _lim), (-_lim, _lim)
p4 = plot(ce; xlims=xlims, ylims=ylims)
scatter!(X̂[1, :], X̂[2, :]; label="X|y=$target", shape=:star5, ms=10, title="ECCo Generator ($(round(faith, digits=5)))", color=3, alpha=0.1)
scatter!(ce.x′[1,:], ce.x′[2,:]; label="Counterfactual", shape=:star1, ms=20, color=4)

plot(p1, p2, p3, p4; size=(1000, 800))
```

## MNIST

``` julia
_nrow = 3
Random.seed!(42)
X, y = TaijaData.load_mnist()
data = CounterfactualData(X, y)

using CounterfactualExplanations.Models: load_mnist_model
using CounterfactualExplanations: JEM
M = load_mnist_model(JEM())
# M = load_mnist_model(MLP())

# Select a factual instance:
target = 3
factual = 8
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Search parameters:
opt = RMSProp(0.01)
conv = GeneratorConditionsConvergence(decision_threshold = 0.9)

# Factual:
factual = convert2image(MNIST, reshape(x, 28, 28))
p1 = plot(factual; title="Factual", axis=([], false))

# Generic Generator:
@info "Generic Generator"
generator = GenericGenerator(opt=opt)
ce = generate_counterfactual(x, target, data, M, generator; convergence=conv, initialization=:identity)
faith = Evaluation.faithfulness(ce; nsamples=_nrow^2, niter_final=10000)
println("Faithfulness: $(round(faith, digits=5))")
plaus = Evaluation.plausibility(ce)
println("Plausibility: $(round(plaus, digits=5))")
img = convert2image(MNIST, reshape(ce.x′, 28, 28))
p2 = plot(img, title="Generic: p(y=$target)=$(round(target_probs(ce)[1], digits=5))", axis=([], false))

# Search:
@info "ECCo Generator"
generator = ECCoGenerator(opt=opt)
ce = generate_counterfactual(x, target, data, M, generator; convergence=conv, initialization=:identity)
faith = Evaluation.faithfulness(ce; nsamples=_nrow^2, niter_final=10000)
println("Faithfulness: $(round(faith, digits=5))")
plaus = Evaluation.plausibility(ce)
println("Plausibility: $(round(plaus, digits=5))")
img = convert2image(MNIST, reshape(ce.x′, 28, 28))
p3 = plot(img, title="ECCo: p(y=$target)=$(round(target_probs(ce)[1], digits=5))", axis=([], false))

plt = plot(p1, p2, p3; size=(750, 200), layout=(1, 3))
display(plt)

X̂ = ce.search[:energy_sampler][ce.target].posterior 
imgs = eachcol(X̂) |>
    X -> reshape.(X, 28, 28) |>
    X -> convert2image.(MNIST, X) |>
    X -> mosaicview(X, nrow=_nrow)
imgs
```
