```@meta
CurrentModule = AlgorithmicRecourse 
```

# Loss functions and gradients

This is a short tutorial on loss functions and gradients typically involved in counterfactual search. It involves more maths than perhaps some of the other tutorials.


```julia
using Zygote, Plots, PlotThemes, LinearAlgebra
theme(:wong)
using Logging
disable_logging(Logging.Info)
```


    LogLevel(1)


## General setup

We begin by restating the general setup for generic counterfactual search. Let $t\in\{0,1\}$ denote the target label, $M$ the model (classifier) and $\underline{x}\in\mathbb{R}^D$ the vector of counterfactual features (we will assume all features are continuous). Then the differentiable optimization problem in algorithmic recourse is generally of the following form

```math
\underline{x} = \arg \min_{\underline{x}}  \ell(M(\underline{x}),t) + \lambda h(\underline{x})
```

where $\ell$ denotes some loss function targeting the deviation between the target label and the predicted label and $h(\cdot)$ acts as a complexity penality generally addressing the *realism* or *cost* of the proposed counterfactual. 

## Loss function $\ell$

Different choices for $\ell$ come to mind, each potentially leading to very different counterfactual outcomes. In practice, $\ell$ is often implemented with respect to the *logits* $a=\mathbf{w}^Tx$ rather than the probabilities $p(\underline{y}=1|\underline{x})=\sigma(a)$ predicted by the classifier. We follow this convention here, but as we shall see *depeding on the label domain this convention does not work well for every type of loss function*. Common choices for $\ell$ in the literature include margin-based loss function like **Hinge** loss and **logit binary crossentropy** (or **log**) loss. Some use distance-based loss such as **mean squared error** loss (MSE).

### Hinge loss

With respect to the logits $a=\mathbf{w}'x$ Hinge loss can be defined as follows

```math
\ell(a,t^*)=(1-a\cdot t^*)_+=\max\{0,1-a\cdot t^*\}
```

where $t^*$ is the target label in $\{-1,1\}$. Since above we defined $t\in\{0,1\}$ we need a mapping $h: \{0,1\} \mapsto \{-1,1\}$. Specifically, we want to plug in $h(t)=t^*$ where $h(\cdot)$ is just the following conditional:

```math
\begin{aligned}
h(t)&=\begin{cases}
-1 && \text{if} && t=0 \\ 1 && \text{if} && t=1
\end{cases}
\end{aligned}
```

Then our loss function as function of $t$ can restated as follows:

```math
\ell(a,t^*)=\ell(a,t)=(1-a\cdot h(t))_+=\max\{0,1-a\cdot h(t)\}
```

The first-order derivative of Hinge loss with respect to the logits $a$ is simply

```math
\begin{aligned}
\ell'(a,t)&=\begin{cases}
-h(t) && \text{if} && a \cdot h(t)<=1 \\ 0 && \text{otherwise.} 
\end{cases}
\end{aligned}
```

In the context of counterfactual search the gradient with respect to the feature vector is then:

```math
\begin{aligned}
&& \nabla_{\underline{x}} \ell(a,t)&= \begin{cases}
-h(t)\mathbf{w} && \text{if} && h(t)\mathbf{w}^T\underline{x}<=1 \\ 0 && \text{otherwise.} 
\end{cases}
\end{aligned}
```

In practice gradients are commonly computed through autodifferentiation. In this tutorial we use the [Zygote.jl](https://github.com/FluxML/Zygote.jl) package which is at the core of [Flux.jl](https://fluxml.ai/Flux.jl/stable/models/basics/), the main deep learning library for Julia.

The side-by-side plot below visualises the loss function and its derivative. The plot further below serves as a simple sanity check to verify that autodifferentiation indeed yields the same result as the closed-form solution for the gradient.


```julia
h(t) = ifelse(t==1,1,-1)
hinge(a,t) = max(0,1-a*h(t))
```


    hinge (generic function with 1 method)



```julia
default(size=(500,500))
a = -2:0.05:2
p1 = plot(a, [hinge(a,1) for a=a], title="Loss, t=1", xlab="logits")
p2 = plot(a, [gradient(hinge,a,1)[1] for a=a], title="Gradient, t=1", xlab="logits")
p3 = plot(a, [hinge(a,0) for a=a], title="Loss, t=0", xlab="logits")
p4 = plot(a, [gradient(hinge,a,0)[1] for a=a], title="Gradient, t=0", xlab="logits")
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
savefig("www/loss_grad_hinge.png")
```

![](www/loss_grad_hinge.png)


```julia
# Just verifying that the formula for the gradient above indeed yields the same result.
function gradient_man(x,w,t)
    𝐠 = ifelse(h(t)*w'x<=1, -h(t)*w, 0)
    return 𝐠
end;
plot(a, [gradient_man(a,1,1) for a=a], legend=:bottomright, label="Manual", title="Gradient", xlab="logits")
scatter!(a, [gradient(hinge,a,1)[1] for a=a], label="Autodiff")
savefig("www/loss_grad_hinge_test.png")
```

![](www/loss_grad_hinge_test.png)

### Logit binary crossentropy loss

Logit binary crossentropy loss loss (sometimes referred to as log loss) is defined as follows:

```math
\begin{aligned}
&& \ell(a,t)&=- \left( t \cdot \log(\sigma(a)) + (1-t) \cdot \log (1-\sigma(a)) \right) \\
\end{aligned}
```

where $\sigma(a)$ is the logit/sigmoid link function.

Once again for the purpose of counter factual search we are interested in the first-order derivative with respect to our feature vector $\underline{x}$. You can verify that the partial derivative with respect to feature $\underline{x}_d$ is as follows:

```math
\begin{aligned}
&& \frac{\partial \ell(a,t)}{\partial \underline{x}_d}&= (\sigma(a) - t) w_d \\
\end{aligned}
```

The gradient just corresponds to the stacked vector of partial derivatives:

```math
\begin{aligned}
&& \nabla_{\underline{x}} \ell(a,t)&= (\sigma(a) - t) \mathbf{w} \\
\end{aligned}
```

As before implementation below is done through autodifferentiation. As before the side-by-side plot shows the resulting loss function and its gradient and the plot further below is a simple sanity check.


```julia
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


    logitbinarycrossentropy (generic function with 1 method)


![](www/loss_grad_log.png)


```julia
p1 = plot(a, [logitbinarycrossentropy(a,1) for a=a], title="Loss, t=1", xlab="logits")
p2 = plot(a, [gradient(logitbinarycrossentropy,a,1)[1] for a=a], title="Gradient, t=1", xlab="logits")
p3 = plot(a, [logitbinarycrossentropy(a,0) for a=a], title="Loss, t=0", xlab="logits")
p4 = plot(a, [gradient(logitbinarycrossentropy,a,0)[1] for a=a], title="Gradient, t=0", xlab="logits")
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
savefig("www/loss_grad_log.png")
```

![](www/loss_grad_log_test.png)


```julia
# Just verifying that the formula for the gradient above indeed yields the same result.
function gradient_man(x,w,y)
    𝐠 = (𝛔(w'x) - y) .* w
    return 𝐠
end;
plot(a, [gradient_man(a,1,1) for a=a], legend=:bottomright, label="Manual", title="Gradient", xlab="logits")
scatter!(a, [gradient(logitbinarycrossentropy,a,1)[1] for a=a], label="Autodiff")
savefig("www/loss_grad_log_test.png")
```

### Mean squared error

Some authors work with distance-based loss functions instead. Since in general we are interested in providing valid recourse, that is counterfactual explanations that indeed lead to the desired label switch, using one of the margin-based loss functions introduced above seems like a more natural choice. Nonetheless, we shall briefly introduce one of the common distance-based loss functions as well. 

The mean squared error for counterfactual search implemented with respect to the logits is simply the squared $\ell^2$ norm between the target label and $a=\mathbf{w}^Tx$:

```math
\begin{aligned}
&& \ell(a,t)&= ||t-a||^2
\end{aligned}
```

The gradient with respect to the vector of features is then:

```math
\begin{aligned}
&& \nabla_{\underline{x}} \ell(a,t)&= 2(a - t) \mathbf{w} \\
\end{aligned}
```

As before implementation and visualizations follow below.


```julia
mse(a,t) = norm(t - a)^2
```


    mse (generic function with 1 method)


**NOTE**: I hinted above that the convention of taking derivatives with respect to logits can go wrong depending on the loss function we choose. The plot below demonstrates this point: for $t=0$ the global minimum of the MSE is of course also at $0$. The implication for counterfactual search is that for $t=0$ the search stops when $\mathbf{w}^T\underline{x}=0$. But at this point $\sigma(\mathbf{w}^T\underline{x})=0.5$, in other words we stop right at the decision boundary, but never cross it. We will see an example of this below. Key takeaway: carefully think about the choice of your loss function and **DON'T** us distance-based loss functions when optimizing with respect to logits.


```julia
p1 = plot(a, [mse(a,1) for a=a], title="Loss, t=1", xlab="logits")
p2 = plot(a, [gradient(mse,a,1)[1] for a=a], title="Gradient, t=1", xlab="logits")
p3 = plot(a, [mse(a,0) for a=a], title="Loss, t=0", xlab="logits")
p4 = plot(a, [gradient(mse,a,0)[1] for a=a], title="Gradient, t=0", xlab="logits")
plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
savefig("www/loss_grad_mse.png")
```

![](www/loss_grad_mse.png)


```julia
# Just verifying that the formula for the gradient above indeed yields the same result.
function gradient_man(x,w,y)
    𝐠 = 2*(w'x - y) .* w
    return 𝐠
end;
plot(a, [gradient_man(a,1,1) for a=a], legend=:bottomright, label="Manual", title="Gradient", xlab="logits")
scatter!(a, [gradient(mse,a,1)[1] for a=a], label="Autodiff")
savefig("www/loss_grad_mse_test.png")
```

![](www/loss_grad_mse_test.png)

## Example in 2D

To understand the properties of the different loss functions we will now look at a toy example in 2D. The code below generates some random features and assigns labels based on a fixed vector of coefficients using the sigmoid function.


```julia
# Some random data:
using Flux
using Random
Random.seed!(1234);
N = 25
w = [1.0 -2.0]# true coefficients
b = 0
X = reshape(randn(2*N),2,N).*1 # random features
y = Int.(round.(Flux.σ.(w*X .+ b))); # label based on sigmoid
```

The plot below shows the samples coloured by label along with the decision boundary. You can think of this as representing the outcome of some automated decision making system. The highlighted sample was chosen to receive algorithmic recourse in the following: we will search for a counterfactual that leads to a label switch.


```julia
# Plot with random sample chose for recourse
function plot_data(;clegend=true,title="",size=1.2.*(400,300))
    x_range = collect(range(minimum(X[1,:]),stop=maximum(X[1,:]),length=50))
    y_range = collect(range(minimum(X[2,:]),stop=maximum(X[2,:]),length=50))
    Z = [Flux.σ.(w * [x,y] .+ b)[1] for x=x_range, y=y_range]
    plt = contourf(
        x_range, y_range, Z', legend=clegend, title=title, size=size, lw=0.1
    )
    scatter!(plt, X[1,reshape(y.==1,25)],X[2,reshape(y.==1,25)],label="y=1",color=1) # features
    scatter!(plt, X[1,reshape(y.==0,25)],X[2,reshape(y.==0,25)],label="y=0",color=0) # features
    Plots.abline!(plt,0.5,b,color="black",label="",lw=2) # decision boundary
    return plt
end

plt = plot_data()
x̅ = X[:,5]
y̅ = round.(Flux.σ.(w*x̅ .+ b))[1]
scatter!(plt,[x̅[1]],[x̅[2]],color=Int.(y̅),markersize=10,label="")
savefig(plt, "www/loss_examlpe.png")
```

![](www/loss_examlpe.png)

Next we will generating recourse using the AlgorithmicRecourse.jl package. First we intantiate our model and based on the assigned label we identify the target (the opposite label).


```julia
using AlgorithmicRecourse
using AlgorithmicRecourse.Models: LogisticModel
𝑴 = LogisticModel(w, [b]);
target = ifelse(y̅==1.0,0.0,1.0)
γ = 0.75
```


    0.75


Now we instantiate different generators for our different loss functions and different choices of $\lambda$. Finally we generate recourse for each of them:


```julia
# Generating recourse
Λ = [0, 1, 5] # varying complexity penalties
losses = [:hinge_loss, :logitbinarycrossentropy, :mse]
recourses = []
for loss in losses
    for λ in Λ
        gen = GenericGenerator(λ,0.1,1e-5,loss,nothing) 
        rec = generate_recourse(gen, x̅, 𝑴, target, γ, T=25)
        recourses = vcat(recourses, (rec=rec, λ=λ, loss=loss))
    end
end
```

The code below plots the resulting counterfactual paths. 

1. **Complexity penalty $\lambda$**: has the expected effect of penalizing *long* counterfactual paths: as the distance between $\overline{x}$ and $\underline{x}$ the penalty exerts more and more pressure on the gradient in the opposite direction $\nabla\ell$. For large choices of $\lambda$ valid recourse is not attainable.
2. **Confidence threshold $\gamma$**: note how for both log loss and hinge loss we overshoot a bit, that is we end up well beyond the decision boundary. This is because above we chose a confidence threshold of $\gamma=0.75$. In the context of recourse this choice matters a lot: we have a longer distance to travel (=higher costs for the individual), but we can be more confident that recourse will remain valid. There is of course an interplay between $\lambda$ and $\gamma$.
3. **The choice of the loss function matters**: the distance-based MSE does **NOT** work without further ajustments when optimizing with respect to logits, as discussed above. 

Overall, in the context of this toy example log **loss arguably generates the most reasonable outcome**: firstly, we can observe that the step size decreases at an increasing rate as the search approaches convergence (which may be desirable); secondly, it appears that increasing $\lambda$ leads to a roughly proportional decrease in the distance of the final counterfactual. This stands in contrast to the outcome for Hinge loss, where increasing $\lambda$ from $0$ to $1$ barely has any effect at all.


```julia
# Plotting
k = length(recourses)
function plot_recourse(rec, idx)
    plt = plot_data(clegend=false, size=(floor(sqrt(k)) * 350, ceil(sqrt(k)) * 350))
    idx_path = minimum([idx, size(rec.rec.path)[1]])
    scatter!(plt, rec.rec.path[1:idx_path,1], rec.rec.path[1:idx_path,2], color=Int(y̅))
    scatter!(plt, [rec.rec.path[idx_path,1]],[rec.rec.path[idx_path,2]],color=Int(y̅),markersize=10)
end
max_path_length = maximum(map(rec -> size(rec.rec.path)[1], recourses))
anim = @animate for i in 1:max_path_length
    plots = map(rec -> plot_recourse(rec, i), recourses);
    plot(plots..., layout = (Int(floor(sqrt(k))), Int(ceil(sqrt(k)))), legend = false, plot_title="Iteration: " * string(i))
end
gif(anim, "www/loss_paths.gif", fps=5);
```

![](www/loss_paths.gif)
