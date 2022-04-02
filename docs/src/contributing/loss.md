``` @meta
CurrentModule = CounterfactualExplanations 
```

# Loss functions

For the computation of loss functions and their gradients we leverage the functionality already implmented in [Flux](https://fluxml.ai/). All of the loss functions from Flux have been imported:

``` julia-repl
julia> names(CounterfactualExplanations.Losses)
19-element Vector{Symbol}:
 :Losses
 :binary_focal_loss
 :binarycrossentropy
 :crossentropy
 :ctc_loss
 :dice_coeff_loss
 :focal_loss
 :hinge_loss
 ⋮
 :logitbinarycrossentropy
 :logitcrossentropy
 :mae
 :mse
 :msle
 :poisson_loss
 :squared_hinge_loss
 :tversky_loss
```

## Classification

For most classification tasks the default `:logitbinarycrossentropy` (binary) and `:logitcrossentropy` should be sufficient. For both choices the package has been tested and works natively. When using other loss functions, some caution is recommended though:

    !!! warning "External loss functions"
        Some margin-based loss functions like hinge loss do not expect inputs in the domain $\mathcal{Y}=\{0,1\}$, but rather $\mathcal{Y}=\{-1,1\}$. In those case one needs to ensure that the training labels $y$ are encoded accordingly. In order to use distance-based loss functions like mean squared error (MSE) loss needs to be computed with respect to probibilities rather than logits. This is currently not supported and we genenerally recommend not to use distance-based loss functions in the classification setting (more on this below).

## Regression

At this point `CounterfactualExplanations.jl` is designed to be used with classification models, since the overwhelming majority of the existing literature on counterfactual explanations is set in this context. By default margin-based loss functions are used and computed with respect to logits (more on this below). To produce counterfactual explanations for regression problems users currently need to binarize the problem: let *t* denote some target value for the continuous dependent variable *y* in the regression context, then we could respecify the dependent variable as *t̃* = 0 for all *y* \< *t* and *t̃* = 1 otherwise. In future work we want to add full support for regression problems.

# Methodological background

This is a short tutorial on loss functions and gradients typically involved in counterfactual search. It involves more maths than perhaps some of the other tutorials.

## General setup

We begin by restating the general setup for generic counterfactual search. Let *t* ∈ {0, 1} denote the target label, *M* the model (classifier) and *x*′ ∈ ℝ^(*D*) the vector of counterfactual features (we will assume all features are continuous). Then the differentiable optimization problem in algorithmic recourse is generally of the following form

``` math
x\prime = \arg \min_{x\prime}  \ell(M(x\prime),t) + \lambda h(x\prime)
```

where ℓ denotes some loss function targeting the deviation between the target label and the predicted label and *h*(⋅) acts as a complexity penality generally addressing the *realism* or *cost* of the proposed counterfactual.

## Loss function ℓ

Different choices for ℓ come to mind, each potentially leading to very different counterfactual outcomes. In practice, ℓ is often implemented with respect to the *logits* *a* = **w**^(*T*)*x* rather than the probabilities *p*(*y*′=1|*x*′) = *σ*(*a*) predicted by the classifier. We follow this convention here, but as we shall see *depeding on the label domain this convention does not work well for every type of loss function*. Common choices for ℓ in the literature include margin-based loss function like **hinge** loss and **logit binary crossentropy** (or **log**) loss. Some use distance-based loss such as **mean squared error** loss (MSE).

### Hinge loss

With respect to the logits *a* = **w**′*x* hinge loss can be defined as follows

``` math
\ell(a,t^*)=(1-a\cdot t^*)_+=\max\{0,1-a\cdot t^*\}
```

where *t*^(\*) is the target label in { − 1, 1}. Since above we defined *t* ∈ {0, 1} we need a mapping *h* : {0, 1} ↦ { − 1, 1}. Specifically, we want to plug in *h*(*t*) = *t*^(\*) where *h*(⋅) is just the following conditional:

``` math
\begin{aligned}
h(t)&=\begin{cases}
-1 && \text{if} && t=0 \\ 1 && \text{if} && t=1
\end{cases}
\end{aligned}
```

Then our loss function as function of *t* can restated as follows:

``` math
\ell(a,t^*)=\ell(a,t)=(1-a\cdot h(t))_+=\max\{0,1-a\cdot h(t)\}
```

The first-order derivative of hinge loss with respect to the logits *a* is simply

``` math
\begin{aligned}
\ell'(a,t)&=\begin{cases}
-h(t) && \text{if} && a \cdot h(t)<=1 \\ 0 && \text{otherwise.} 
\end{cases}
\end{aligned}
```

In the context of counterfactual search the gradient with respect to the feature vector is then:

``` math
\begin{aligned}
&& \nabla_{x\prime} \ell(a,t)&= \begin{cases}
-h(t)\mathbf{w} && \text{if} && h(t)\mathbf{w}^Tx\prime<=1 \\ 0 && \text{otherwise.} 
\end{cases}
\end{aligned}
```

In practice gradients are commonly computed through autodifferentiation. In this tutorial we use the [Zygote.jl](https://github.com/FluxML/Zygote.jl) package which is at the core of [Flux.jl](https://fluxml.ai/Flux.jl/stable/models/basics/), the main deep learning library for Julia.

The side-by-side plot below visualises the loss function and its derivative. The plot further below serves as a simple sanity check to verify that autodifferentiation indeed yields the same result as the closed-form solution for the gradient.

``` julia
h(t) = ifelse(t==1,1,-1)
hinge(a,t) = max(0,1-a*h(t))
```

``` julia
default(size=(500,500))
a = -2:0.05:2
p1 = plot(a, [hinge(a,1) for a=a], title="Loss, t=1", xlab="logits")
p2 = plot(a, [gradient(hinge,a,1)[1] for a=a], title="Gradient, t=1", xlab="logits")
p3 = plot(a, [hinge(a,0) for a=a], title="Loss, t=0", xlab="logits")
p4 = plot(a, [gradient(hinge,a,0)[1] for a=a], title="Gradient, t=0", xlab="logits")
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
savefig(joinpath(www_path, "loss_grad_hinge.png"))
```

![](www/loss_grad_hinge.png)

``` julia
# Just verifying that the formula for the gradient above indeed yields the same result.
function gradient_man(x,w,t)
    𝐠 = ifelse(h(t)*w'x<=1, -h(t)*w, 0)
    return 𝐠
end;
plot(a, [gradient_man(a,1,1) for a=a], legend=:bottomright, label="Manual", title="Gradient", xlab="logits")
scatter!(a, [gradient(hinge,a,1)[1] for a=a], label="Autodiff")
savefig(joinpath(www_path, "loss_grad_hinge_test.png"))
```

![](www/loss_grad_hinge_test.png)

### Logit binary crossentropy loss

Logit binary crossentropy loss loss (sometimes referred to as log loss) is defined as follows:

``` math
\begin{aligned}
&& \ell(a,t)&=- \left( t \cdot \log(\sigma(a)) + (1-t) \cdot \log (1-\sigma(a)) \right) \\
\end{aligned}
```

where *σ*(*a*) is the logit/sigmoid link function.

Once again for the purpose of counter factual search we are interested in the first-order derivative with respect to our feature vector *x*′. You can verify that the partial derivative with respect to feature *x*′_(*d*) is as follows:

``` math
\begin{aligned}
&& \frac{\partial \ell(a,t)}{\partial x\prime_d}&= (\sigma(a) - t) w_d \\
\end{aligned}
```

The gradient just corresponds to the stacked vector of partial derivatives:

``` math
\begin{aligned}
&& \nabla_{x\prime} \ell(a,t)&= (\sigma(a) - t) \mathbf{w} \\
\end{aligned}
```

As before implementation below is done through autodifferentiation. As before the side-by-side plot shows the resulting loss function and its gradient and the plot further below is a simple sanity check.

``` julia
# sigmoid function:
function 𝛔(a)
    trunc = 8.0 # truncation to avoid numerical over/underflow
    a = clamp.(a,-trunc,trunc)
    p = exp.(a)
    p = p ./ (1 .+ p)
    return p
end

# Logit binary crossentropy:
logitbinarycrossentropy(a, t) = - (t * log(𝛔(a)) + (1-t) * log(1-𝛔(a)))
```

![](www/loss_grad_log.png)

``` julia
p1 = plot(a, [logitbinarycrossentropy(a,1) for a=a], title="Loss, t=1", xlab="logits")
p2 = plot(a, [gradient(logitbinarycrossentropy,a,1)[1] for a=a], title="Gradient, t=1", xlab="logits")
p3 = plot(a, [logitbinarycrossentropy(a,0) for a=a], title="Loss, t=0", xlab="logits")
p4 = plot(a, [gradient(logitbinarycrossentropy,a,0)[1] for a=a], title="Gradient, t=0", xlab="logits")
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
savefig(joinpath(www_path, "loss_grad_log.png"))
```

![](www/loss_grad_log_test.png)

``` julia
# Just verifying that the formula for the gradient above indeed yields the same result.
function gradient_man(x,w,y)
    𝐠 = (𝛔(w'x) - y) .* w
    return 𝐠
end;
plot(a, [gradient_man(a,1,1) for a=a], legend=:bottomright, label="Manual", title="Gradient", xlab="logits")
scatter!(a, [gradient(logitbinarycrossentropy,a,1)[1] for a=a], label="Autodiff")
savefig(joinpath(www_path, "loss_grad_log_test.png"))
```

### Mean squared error

Some authors work with distance-based loss functions instead. Since in general we are interested in providing valid recourse, that is counterfactual explanations that indeed lead to the desired label switch, using one of the margin-based loss functions introduced above seems like a more natural choice. Nonetheless, we shall briefly introduce one of the common distance-based loss functions as well.

The mean squared error for counterfactual search implemented with respect to the logits is simply the squared ℓ² norm between the target label and *a* = **w**^(*T*)*x*:

``` math
\begin{aligned}
&& \ell(a,t)&= ||t-a||^2
\end{aligned}
```

The gradient with respect to the vector of features is then:

``` math
\begin{aligned}
&& \nabla_{x\prime} \ell(a,t)&= 2(a - t) \mathbf{w} \\
\end{aligned}
```

As before implementation and visualizations follow below.

``` julia
mse(a,t) = norm(t - a)^2
```

**NOTE**: I hinted above that the convention of taking derivatives with respect to logits can go wrong depending on the loss function we choose. The plot below demonstrates this point: for *t* = 0 the global minimum of the MSE is of course also at 0. The implication for counterfactual search is that for *t* = 0 the search stops when **w**^(*T*)*x*′ = 0. But at this point *σ*(**w**^(*T*)*x*′) = 0.5, in other words we stop right at the decision boundary, but never cross it. We will see an example of this below. Key takeaway: carefully think about the choice of your loss function and **DON’T** use distance-based loss functions when optimizing with respect to logits.

``` julia
p1 = plot(a, [mse(a,1) for a=a], title="Loss, t=1", xlab="logits")
p2 = plot(a, [gradient(mse,a,1)[1] for a=a], title="Gradient, t=1", xlab="logits")
p3 = plot(a, [mse(a,0) for a=a], title="Loss, t=0", xlab="logits")
p4 = plot(a, [gradient(mse,a,0)[1] for a=a], title="Gradient, t=0", xlab="logits")
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
savefig(joinpath(www_path, "loss_grad_mse.png"))
```

![](www/loss_grad_mse.png)

``` julia
# Just verifying that the formula for the gradient above indeed yields the same result.
function gradient_man(x,w,y)
    𝐠 = 2*(w'x - y) .* w
    return 𝐠
end;
plot(a, [gradient_man(a,1,1) for a=a], legend=:bottomright, label="Manual", title="Gradient", xlab="logits")
scatter!(a, [gradient(mse,a,1)[1] for a=a], label="Autodiff")
savefig(joinpath(www_path, "loss_grad_mse_test.png"))
```

![](www/loss_grad_mse_test.png)

## Example in 2D

To understand the properties of the different loss functions we will now look at a toy example in 2D. The code below generates some random features and assigns labels based on a fixed vector of coefficients using the sigmoid function.

``` julia
# Some random data:
using Flux, Random, CounterfactualExplanations.Data
Random.seed!(1234)
N = 25
w = [1.0 1.0]# true coefficients
b = 0
xs, ys = Data.toy_data_linear(N)
X = hcat(xs...)
counterfactual_data = CounterfactualData(X,ys')
```

The plot below shows the samples coloured by label along with the decision boundary. You can think of this as representing the outcome of some automated decision making system. The highlighted sample was chosen to receive algorithmic recourse in the following: we will search for a counterfactual that leads to a label switch.

``` julia
using CounterfactualExplanations
using CounterfactualExplanations.Models: LogisticModel
M = LogisticModel(w, [b])

Random.seed!(1234)
x = select_factual(counterfactual_data,rand(1:size(X)[2]))
y = round(probs(M, x)[1])
target = ifelse(y==1.0,0.0,1.0)
γ = 0.75

# Plot with random sample chose for recourse
plt = plot_contour(X',ys,M)
scatter!(plt,[x[1]],[x[2]],ms=10,label="", color=Int(y))
Plots.abline!(plt,-w[2]/w[1],b,color="black",label="",lw=2)
savefig(joinpath(www_path, "loss_example.png"))
```

![](www/loss_example.png)

Now we instantiate different generators for our different loss functions and different choices of *λ*. Finally we generate recourse for each of them:

``` julia
# Generating recourse
Λ = [0.0, 1.0, 5.0] # varying complexity penalties
losses = [:hinge_loss, :logitbinarycrossentropy, :mse]
counterfactuals = []
for loss in losses
    for λ in Λ
        generator = GenericGenerator(;loss=loss,λ=λ) 
        t = loss == :hinge_loss ? h(target) : target # mapping for hinge loss
        counterfactual = generate_counterfactual(x, t, counterfactual_data, M, generator; γ=γ, T=50)
        counterfactuals = vcat(counterfactuals, counterfactual)
    end
end
```

The code below plots the resulting counterfactual paths.

1.  **Complexity penalty *λ***: has the expected effect of penalizing *long* counterfactual paths: as the distance between *x* and *x*′ the penalty exerts more and more pressure on the gradient in the opposite direction ∇ℓ. For large choices of *λ* valid recourse is not attainable.
2.  **Confidence threshold *γ***: note how for both log loss and hinge loss we overshoot a bit, that is we end up well beyond the decision boundary. This is because above we chose a confidence threshold of *γ* = 0.75. In the context of recourse this choice matters a lot: we have a longer distance to travel (=higher costs for the individual), but we can be more confident that recourse will remain valid. There is of course an interplay between *λ* and *γ*.
3.  **The choice of the loss function matters**: the distance-based MSE does **NOT** work without further ajustments when optimizing with respect to logits, as discussed above.

Overall, in the context of this toy example log **loss arguably generates the most reasonable outcome**: firstly, we can observe that the step size decreases at an increasing rate as the search approaches convergence (which may be desirable); secondly, it appears that increasing *λ* leads to a roughly proportional decrease in the distance of the final counterfactual. This stands in contrast to the outcome for hinge loss, where increasing *λ* from 0 to 1 barely has any effect at all.

``` julia
# Plotting
k = length(counterfactuals)
function plot_recourse(counterfactual, t)
    l = string(counterfactual.generator.loss)
    l = l[1:minimum([5,length(l)])]
    λ = string(counterfactual.generator.λ)
    plt = plot_contour(X',ys,M;colorbar=false,title="Loss: $(l), λ: $(λ)")
    plt = plot(plt, size=(floor(√(k)) * 350, ceil(√(k)) * 350))
    Plots.abline!(plt,-w[2]/w[1],b,color="black",label="",lw=2)
    t = minimum([t, total_steps(counterfactual)])
    scatter!(plt, hcat(path(counterfactual)[1:t]...)[1,:], hcat(path(counterfactual)[1:t]...)[2,:], ms=10, color=Int(y), label="")
    return plt
end
max_path_length = maximum(map(counterfactual -> total_steps(counterfactual), counterfactuals))
anim = @animate for i in 1:max_path_length
    plots = map(counterfactual -> plot_recourse(counterfactual, i), counterfactuals)
    plot(plots..., layout = (Int(floor(√(k))), Int(ceil(√(k)))), legend = false, plot_title="Iteration: " * string(i))
end
gif(anim, joinpath(www_path, "loss_paths.gif"), fps=5)
```

![](www/loss_paths.gif)
