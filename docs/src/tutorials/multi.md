```@meta
CurrentModule = CounterfactualExplanations 
```

# Recourse for multi-class targets


```julia
using Flux, Random, Plots, PlotThemes, CounterfactualExplanations, Statistics
theme(:wong)
using Logging
disable_logging(Logging.Info)
```


    LogLevel(1)



```julia
x, y = toy_data_multi()
X = hcat(x...)
y_train = Flux.onehotbatch(y, unique(y))
y_train = Flux.unstack(y_train',1)
plt = plot()
plt = plot_data!(plt,X',y);
savefig(plt, "www/multi_samples.png")
```

![](www/multi_samples.png)

## Classifier


```julia
n_hidden = 32
out_dim = length(unique(y))
kw = (output_dim=out_dim, dropout=true)
nn = build_model(;kw...)
loss(x, y) = Flux.Losses.logitcrossentropy(nn(x), y)
ps = Flux.params(nn)
data = zip(x,y_train);
```


```julia
using Flux.Optimise: update!, ADAM
opt = ADAM()
epochs = 10
avg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))
show_every = epochs/10

for epoch = 1:epochs
  for d in data
    gs = gradient(params(nn)) do
      l = loss(d...)
    end
    update!(opt, params(nn), gs)
  end
  if epoch % show_every == 0
    println("Epoch " * string(epoch))
    @show avg_loss(data)
  end
end
```

    Epoch 1
    avg_loss(data) = 0.9255239012607264
    Epoch 2
    avg_loss(data) = 0.3593051233387213
    Epoch 3
    avg_loss(data) = 0.18421732400655624
    Epoch 4
    avg_loss(data) = 0.10711486082055025
    Epoch 5
    avg_loss(data) = 0.07511142481836484
    Epoch 6
    avg_loss(data) = 0.0575109613420611
    Epoch 7
    avg_loss(data) = 0.0424017922374355
    Epoch 8
    avg_loss(data) = 0.03331096899975358
    Epoch 9
    avg_loss(data) = 0.027016712426665555
    Epoch 10
    avg_loss(data) = 0.02219848870252177



```julia
using CounterfactualExplanations, CounterfactualExplanations.Models
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

# Step 1)
struct NeuralNetwork <: Models.AbstractFittedModel
    nn::Any
end

# Step 2)
logits(𝑴::NeuralNetwork, X::AbstractArray) = 𝑴.nn(X)
probs(𝑴::NeuralNetwork, X::AbstractArray)= softmax(logits(𝑴, X))
𝑴 = NeuralNetwork(nn);
```


```julia
plt = plot_contour_multi(X',y,𝑴);
savefig(plt, "www/multi_contour.png")
```

![](www/multi_contour.png)


```julia
# Randomly selected factual:
Random.seed!(42);
x̅ = X[:,rand(1:size(X)[2])]
y̅ = Flux.onecold(probs(𝑴, x̅),unique(y))
target = rand(unique(y)[1:end .!= y̅]) # opposite label as target
γ = 0.75
# Define AbstractGenerator:
generator = GenericGenerator(0.1,0.1,1e-5,:logitcrossentropy,nothing)
# Generate recourse:
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ); # generate recourse
```


```julia
T = size(recourse.path)[1]
X_path = reduce(hcat,recourse.path)
ŷ = CounterfactualExplanations.target_probs(probs(recourse.𝑴, X_path),target)
p1 = plot_contour(X',y,𝑴;clegend=false, title="MLP")
anim = @animate for t in 1:T
    scatter!(p1, [recourse.path[t][1]], [recourse.path[t][2]], ms=5, color=Int(y̅), label="")
    p2 = plot(1:t, ŷ[1:t], xlim=(0,T), ylim=(0, 1), label="p(y̲=" * string(target) * ")", title="Validity", lc=:black)
    Plots.abline!(p2,0,γ,label="threshold γ", ls=:dash) # decision boundary
    plot(p1,p2,size=(800,400))
end
gif(anim, "www/multi_generic_recourse.gif", fps=5);
```

![](www/multi_generic_recourse.gif)

## Deep ensemble


```julia
𝓜 = build_ensemble(5;kw=(output_dim=out_dim,));
```


```julia
using CounterfactualExplanations: forward
𝓜, anim = forward(𝓜, data, opt, n_epochs=epochs, plot_every=1); # fit the ensemble
gif(anim, "www/multi_ensemble_loss.gif", fps=10);
```

![](www/multi_ensemble_loss.gif)


```julia
# Step 1)
struct FittedEnsemble <: Models.AbstractFittedModel
    𝓜::AbstractArray
end

# Step 2)
using Statistics
logits(𝑴::FittedEnsemble, X::AbstractArray) = mean(Flux.stack([nn(X) for nn in 𝑴.𝓜],3), dims=3)
probs(𝑴::FittedEnsemble, X::AbstractArray) = mean(Flux.stack([softmax(nn(X)) for nn in 𝑴.𝓜],3),dims=3)

𝑴=FittedEnsemble(𝓜);
```


```julia
plt = plot_contour_multi(X',y,𝑴);
savefig(plt, "www/multi_ensemble_contour.png")
```

![](www/multi_ensemble_contour.png)


```julia
generator = GreedyGenerator(0.25,20,:logitcrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ); # generate recourse
```


```julia
T = size(recourse.path)[1]
X_path = reduce(hcat,recourse.path)
ŷ = CounterfactualExplanations.target_probs(probs(recourse.𝑴, X_path),target)
p1 = plot_contour(X',y,𝑴;clegend=false, title="Deep ensemble")
anim = @animate for t in 1:T
    scatter!(p1, [recourse.path[t][1]], [recourse.path[t][2]], ms=5, color=Int(y̅), label="")
    p2 = plot(1:t, ŷ[1:t], xlim=(0,T), ylim=(0, 1), label="p(y̲=" * string(target) * ")", title="Validity", lc=:black)
    Plots.abline!(p2,0,γ,label="threshold γ", ls=:dash) # decision boundary
    plot(p1,p2,size=(800,400))
end
gif(anim, "www/multi_greedy_recourse.gif", fps=5);
```

![](www/multi_greedy_recourse.gif)
