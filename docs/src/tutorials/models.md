```@meta
CurrentModule = AlgorithmicRecourse 
```

# Models

## Default models

There are currently structures for two default models that can be used with AlgorithmicRecourse.jl:

1. [`LogisticModel(w::AbstractArray,b::AbstractArray)`](@ref)
2. [`BayesianLogisticModel(μ::AbstractArray,Σ::AbstractArray)`](@ref)

Both take sets of estimated parameters at the point of instantiation: the constructors will not fit a model for you, but assume that you have already estimated the respective model yourself and have access to its parameter estimates. Based on the supplied parameters methods to predict logits and probabilities are already implemented and used in the counterfactual search. 

For the simple logistic regression model logits are computed as $a=Xw + b$ and probabilities are simply $\sigma(a)$. For the Bayesian logistic regression model logits are computed as $X\mu$ and the predictive posterior is computed through Laplace approximation.

## Custom models

Apart from the default models you can use any arbitrary (differentiable) model and generate recourse in the same way as before. Only two steps are necessary to make your own model compatible with AlgorithmicRecourse.jl:

1. The model needs to be declared as a subtype of `AlgorithmicRecourse.Models.FittedModel`.
2. You need to extend the functions `AlgorithmicRecourse.Models.logits` and `AlgorithmicRecourse.Models.probs` to accept your custom model.

Below we will go through a simple example to see how this can be done in practice. 

### Neural network

In this example we will build a simple artificial neural network using [Flux.jl](https://fluxml.ai/) for a binary classification task.


```julia
# Import libraries.
using Flux, Plots, Random, PlotThemes, Statistics
theme(:juno)
using Logging
disable_logging(Logging.Info)
```


    LogLevel(1)


First we generate some toy data below. The code that generates this data was borrowed from a great tutorial about Bayesian neural networks provided by [Turing.jl](https://turing.ml/dev/), which you may find [here](https://turing.ml/dev/tutorials/03-bayesian-neural-network/). 

The plot below shows the generated samples in the 2D feature space where colours indicate the associated labels. Clearly this data is not linearly separable and the default `LogisticModel` would be ill suited for this classification task.


```julia
# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))

x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5; 
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s]
X = hcat(xs...) # bring into tabular format
ts = [ones(2*M); zeros(2*M)]

# Plot data points.
function plot_data()
    # x1 = map(e -> e[1], xt1s)
    # y1 = map(e -> e[2], xt1s)
    # x2 = map(e -> e[1], xt0s)
    # y2 = map(e -> e[2], xt0s)

    # Plots.scatter(x1,y1, color=1, clim = (0,1), label="y=1")
    # Plots.scatter!(x2,y2, color=0, clim = (0,1), label="y=0")
    Plots.scatter(X[1,:],X[2,:],color=Int.(ts), clim = (0,1), legend=false)
end

plt = plot_data();
savefig(plt, "www/models_samples.png")
```

![](www/models_samples.png)

#### Training the model

Instead we will build a simple artificial neural network `nn` with one hidden layer. For additional resources on how to do deep learning with [Flux.jl](https://fluxml.ai/) just have a look at their documentation. 


```julia
function build_model(;input_dim=2,n_hidden=32,output_dim=1)
    
    # Params:
    W₁ = input_dim
    b₁ = n_hidden
    W₀ = n_hidden
    b₀ = output_dim
    
    nn = Chain(
        Dense(W₁, b₁, σ),
        Dense(W₀, b₀))  

    return nn

end
nn = build_model()
loss(x, y) = Flux.Losses.logitbinarycrossentropy(nn(x), y)
ps = Flux.params(nn)
data = zip(xs,ts);
```

The code below trains the neural network for the task at hand. The plot shows the (training) loss over time. Note that normally we would be interested in loss with respect to a validation data set. But since we are primarily interested in generated recourse for a trained classifier, here we will just keep things very simple.


```julia
using Flux.Optimise: update!, ADAM
opt = ADAM()
epochs = 200
avg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))

using Plots
anim = Animation()
plt = plot(ylim=(0,avg_loss(data)), xlim=(0,epochs), legend=false, xlab="Epoch")
avg_l = []

for epoch = 1:epochs
  for d in data
    gs = gradient(params(nn)) do
      l = loss(d...)
    end
    update!(opt, params(nn), gs)
  end
  avg_l = vcat(avg_l,avg_loss(data))
  plot!(plt, avg_l, color=1, title="Average (training) loss")
  frame(anim, plt)
end

gif(anim, "www/models_loss.gif");
```

![](www/models_loss.gif)

The plot below shows the predicted probabilities in the feature domain. Evidently our simple neural network is doing very well on the training data, as explected. 


```julia
# Plot the posterior distribution with a contour plot.
x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
Z = [σ.(nn([x, y]))[1] for x=x_range, y=y_range]
function plot_contour(;clegend=true, title="")
    plt = contourf(x_range, y_range, Z, color=:viridis, legend=clegend, title=title)
    scatter!(plt,X[1,:],X[2,:],color=Int.(ts), clim = (0,1), legend=false)
end
plt = plot_contour();
savefig(plt, "www/models_contour.png")
```

![](www/models_contour.png)

#### Generating recourse

Now it's game time: we have a fitted model $M: \mathcal{X} \mapsto y$ and are interested in generating recourse for some individual $\overline{x}\in\mathcal{X}$. As mentioned above we need to do a bit more work to prepare the model to be used by AlgorithmicRecourse.jl. 

The code below takes care of all of that: in step 1) it declares our model as a subtype of `Models.FittedModel` and in step 2) it just extends the two functions. 


```julia
using AlgorithmicRecourse, AlgorithmicRecourse.Models
import AlgorithmicRecourse.Models: logits, probs # import functions in order to extend

# Step 1)
struct NeuralNetwork <: Models.FittedModel
    nn::Any
end

# Step 2)
logits(𝑴::NeuralNetwork, X::AbstractArray) = 𝑴.nn(X)
probs(𝑴::NeuralNetwork, X::AbstractArray)= σ.(logits(𝑴, X))
𝑴 = NeuralNetwork(nn)
```


    NeuralNetwork(Chain(Dense(2, 32, σ), Dense(32, 1)))


Now we just select a random sample from our data and based on its current label we set as our target the opposite label and desired threshold for the predicted probability.


```julia
using Random
Random.seed!(1234)
x̅ = X[:,rand(1:size(X)[2])]
y̅ = round(probs(𝑴, x̅)[1])
target = ifelse(y̅==1.0,0.0,1.0) # opposite label as target
γ = ifelse(target==1.0,0.75,0.25); # desired threshold based on target
```

Then finally we use the `GenericGenerator` to generate recourse. The plot further below shows the resulting counterfactual path.


```julia
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)
recourse = generate_recourse(generator, x̅, 𝑴, target, γ); # generate recourse
```


```julia
T = size(recourse.path)[1]
ŷ = probs(recourse.𝑴, recourse.path')
p1 = plot_contour(;clegend=false, title="Neural network")
anim = @animate for t in 1:T
    scatter!(p1, [recourse.path[t,1]], [recourse.path[t,2]], ms=5, color=Int(y̅))
    p2 = plot(1:t, ŷ[1:t], xlim=(0,T), ylim=(0, 1), label="p(y̲=1)", title="Validity")
    Plots.abline!(p2,0,γ,label="threshold γ") # decision boundary
    plot(p1,p2,size=(800,400))
end
gif(anim, "www/models_generic_recourse.gif", fps=5);
```

![](www/models_generic_recourse.gif)
