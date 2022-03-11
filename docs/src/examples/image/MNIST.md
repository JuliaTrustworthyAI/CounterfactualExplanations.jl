``` @meta
CurrentModule = CounterfactualExplanations 
```

# MNIST

In this examples we will see how different counterfactual generators can be used to explain deep learning models for image classification. In particular, we will look at MNIST data and visually inspect how the different generators perturb images of handwritten digits in order to change the predicted label to a target label. [Figure 1](#fig-digits) shows a random sample of handwritten digits.

<div class="cell" execution_count="37">

``` julia
using CounterfactualExplanations, Plots, MLDatasets
using MLDatasets.MNIST: convert2image
using BSON: @save, @load
```

</div>

<div class="cell" execution_count="53">

``` julia
train_x, train_y = MNIST.traindata()
input_dim = prod(size(train_x[:,:,1]))
using Images, Random, StatsBase
Random.seed!(1234)
n_samples = 10
samples = train_x[:,:,sample(1:end, n_samples, replace=false)]
mosaicview([convert2image(samples[:,:,i]) for i ∈ 1:n_samples]...,ncol=n_samples)
```

<div class="cell-output-display">

<figure>
<img src="MNIST_files/figure-gfm/fig-digits-output-1.png" id="fig-digits" alt="Figure 1: A random sample of handwritten digits." />
<figcaption aria-hidden="true">Figure 1: A random sample of handwritten digits.</figcaption>
</figure>

</div>

</div>

## Pre-trained classifiers

<div class="cell" execution_count="54">

``` julia
using Flux
using CounterfactualExplanations.Data: mnist_data, mnist_model, mnist_ensemble
x,y,data = getindex.(Ref(mnist_data()), ("x", "y", "data"))
model = mnist_model()
𝓜 = mnist_ensemble();
```

</div>

#### MLP

<div class="cell" execution_count="55">

``` julia
using CounterfactualExplanations, CounterfactualExplanations.Models
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

# Step 1)
struct NeuralNetwork <: Models.FittedModel
    nn::Any
end

# Step 2)
logits(𝑴::NeuralNetwork, X::AbstractArray) = 𝑴.nn(X)
probs(𝑴::NeuralNetwork, X::AbstractArray)= softmax(logits(𝑴, X))
𝑴 = NeuralNetwork(model);
```

</div>

#### Deep ensemble

<div class="cell" execution_count="56">

``` julia
# Step 1)
struct FittedEnsemble <: Models.FittedModel
    𝓜::AbstractArray
end

# Step 2)
using Statistics
logits(𝑴::FittedEnsemble, X::AbstractArray) = mean(Flux.stack([nn(X) for nn in 𝑴.𝓜],3), dims=3)
probs(𝑴::FittedEnsemble, X::AbstractArray) = mean(Flux.stack([softmax(nn(X)) for nn in 𝑴.𝓜],3),dims=3)

𝑴_ensemble=FittedEnsemble(𝓜);
```

</div>

## Recourse

We will use four different approaches to generate recourse:

1.  Wachter for the MLP
2.  Greedy approach for the MLP
3.  Wachter for the deep ensemble
4.  Greedy approach for the deep ensemble (Schut et al.)

<div class="cell" execution_count="57">

``` julia
# Randomly selected factual:
Random.seed!(1234);
x̅ = Flux.unsqueeze(x[:,rand(1:size(x)[2])],2)
target = 5
γ = 0.95
img = convert2image(reshape(x̅,Int(sqrt(input_dim)),Int(sqrt(input_dim))))
plt_orig = plot(img, title="Original", axis=nothing)
```

<div class="cell-output-display">

![](MNIST_files/figure-gfm/cell-7-output-1.svg)

</div>

</div>

<div class="cell" execution_count="134">

``` julia
# Generic - MLP
generator = GenericGenerator(0.1,0.1,1e-5,:logitcrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ; feasible_range=(0.0,1.0)) # generate recourse
img = convert2image(reshape(recourse.x̲,Int(sqrt(input_dim)),Int(sqrt(input_dim))))
plt_wachter = plot(img, title="MLP - Wachter");
```

</div>

<div class="cell" execution_count="135">

``` julia
# Greedy - MLP
generator = GreedyGenerator(0.1,15,:logitcrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ; feasible_range=(0.0,1.0)) # generate recourse
img = convert2image(reshape(recourse.x̲,Int(sqrt(input_dim)),Int(sqrt(input_dim))))
plt_greedy = plot(img, title="MLP - Greedy");
```

</div>

<div class="cell" execution_count="136">

``` julia
# Generic - Deep Ensemble
generator = GenericGenerator(0.1,0.1,1e-5,:logitcrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴_ensemble, target, γ; feasible_range=(0.0,1.0)) # generate recourse
img = convert2image(reshape(recourse.x̲,Int(sqrt(input_dim)),Int(sqrt(input_dim))))
plt_wachter_de = plot(img, title="Ensemble - Wachter");
```

</div>

<div class="cell" execution_count="137">

``` julia
# Greedy
generator = GreedyGenerator(0.1,15,:logitcrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴_ensemble, target, γ; feasible_range=(0.0,1.0)) # generate recourse
img = convert2image(reshape(recourse.x̲,Int(sqrt(input_dim)),Int(sqrt(input_dim))))
plt_greedy_de = plot(img, title="Ensemble - Greedy");
```

</div>

<div class="cell" execution_count="138">

``` julia
plt_list = [plt_orig, plt_wachter, plt_greedy, plt_wachter_de, plt_greedy_de]
plt = plot(plt_list...,layout=(1,length(plt_list)),axis=nothing, size=(1200,240))
savefig(plt, "www/MNIST_9to4.png")
```

</div>

![](www/MNIST_9to4.png)

### Multiple

<div class="cell" execution_count="109">

``` julia
using Random

# Single:
function from_digit_to_digit(from, to, generator, model; γ=0.95, x=x, y=y, seed=1234, T=1000)

    Random.seed!(seed)

    candidates = findall(onecold(y,0:9).==from)
    x̅ = Flux.unsqueeze(x[:,rand(candidates)],2)
    target = to + 1
    recourse = generate_counterfactual(generator, x̅, model, target, γ; feasible_range=(0.0,1.0), T=T)

    return recourse
end

# Multiple:
function from_digit_to_digit(from, to, generator::Dict, model::Dict; γ=0.95, x=x, y=y, seed=1234, T=1000)

    Random.seed!(seed)

    candidates = findall(onecold(y,0:9).==from)
    x̅ = Flux.unsqueeze(x[:,rand(candidates)],2)
    target = to + 1
    recourses = Dict()

    for (k_gen,v_gen) ∈ generators
        for (k_mod,v_mod) ∈ models 
            k = k_mod * " - " * k_gen
            recourses[k] = generate_counterfactual(v_gen, x̅, v_mod, target, γ; feasible_range=(0.0,1.0), T=T)
        end
    end

    return recourses
end

```

<div class="cell-output-display">

    from_digit_to_digit (generic function with 2 methods)

</div>

</div>

<div class="cell" execution_count="119">

``` julia
generators = Dict("Wachter" => GenericGenerator(0.1,1,1e-5,:logitcrossentropy,nothing),"Greedy" => GreedyGenerator(0.1,15,:logitcrossentropy,nothing))
models = Dict("MLP" => 𝑴, "Ensemble" => 𝑴_ensemble);
```

</div>

<div class="cell" execution_count="112">

``` julia
from = 3
to = 8
recourses = from_digit_to_digit(from,to,generators,models)
plts =  first(values(recourses)).x̅ |> x -> plot(convert2image(reshape(x,Int(sqrt(input_dim)),Int(sqrt(input_dim)))),title="Original")
plts = vcat(plts, [plot(convert2image(reshape(v.x̲,Int(sqrt(input_dim)),Int(sqrt(input_dim)))),title=k) for (k,v) in recourses])
plt = plot(plts...,layout=(1,length(plts)),axis=nothing, size=(1200,240))
savefig(plt, "www/MNIST_$(from)to$(to).png")
```

</div>

<div class="cell" execution_count="139">

``` julia
from = 7
to = 2
recourses = from_digit_to_digit(from,to,generators,models)
plts =  first(values(recourses)).x̅ |> x -> plot(convert2image(reshape(x,Int(sqrt(input_dim)),Int(sqrt(input_dim)))),title="Original")
plts = vcat(plts, [plot(convert2image(reshape(v.x̲,Int(sqrt(input_dim)),Int(sqrt(input_dim)))),title=k) for (k,v) in recourses])
plt = plot(plts...,layout=(1,length(plts)),axis=nothing, size=(1200,240))
savefig(plt, "www/MNIST_$(from)to$(to).png")
```

</div>

<div class="cell" execution_count="140">

``` julia
from = 1
to = 7
recourses = from_digit_to_digit(from,to,generators,models)
plts =  first(values(recourses)).x̅ |> x -> plot(convert2image(reshape(x,Int(sqrt(input_dim)),Int(sqrt(input_dim)))),title="Original")
plts = vcat(plts, [plot(convert2image(reshape(v.x̲,Int(sqrt(input_dim)),Int(sqrt(input_dim)))),title=k) for (k,v) in recourses])
plt = plot(plts...,layout=(1,length(plts)),axis=nothing, size=(1200,240))
savefig(plt, "www/MNIST_$(from)to$(to).png")
```

</div>

<div class="cell" execution_count="97">

``` julia
from = 9
recourses = map(d -> from_digit_to_digit(from,d,GreedyGenerator(0.1,15,:logitcrossentropy,nothing),𝑴_ensemble;T=2500),filter(x -> x!=from, Vector(0:9)))
plts = [plot(convert2image(reshape(rec.x̲,Int(sqrt(input_dim)),Int(sqrt(input_dim)))),title=rec.target-1) for rec in recourses]
plot(plts...,layout=(1,length(plts)),axis=nothing,size=(1200,500))
```

<div class="cell-output-display">

![](MNIST_files/figure-gfm/cell-18-output-1.svg)

</div>

</div>
