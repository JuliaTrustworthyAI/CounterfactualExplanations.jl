

``` @meta
CurrentModule = CounterfactualExplanations 
```

# `DiCEGenerator`

The `DiCEGenerator` can be used to generate multiple diverse counterfactuals for a single factual.

## Description

Counterfactual Explanations are not unique and there are therefore many different ways through which valid counterfactuals can be generated. In the context of Algorithmic Recourse this can be leveraged to offer individuals not one, but possibly many different ways to change a negative outcome into a positive one. One might argue that it makes sense for those different options to be as diverse as possible. This idea is at the core of **DiCE**, a counterfactual generator introduce by Mothilal, Sharma, and Tan (2020) that generate a diverse set of counterfactual explanations.

### Defining Diversity

To ensure that the generated counterfactuals are diverse, Mothilal, Sharma, and Tan (2020) add a diversity constraint to the counterfactual search objective. In particular, diversity is explicitly proxied via Determinantal Point Processes (DDP).

We can implement DDP in Julia as follows:\[1\]

``` julia
using LinearAlgebra
function ddp_diversity(X::AbstractArray{<:Real, 3})
    xs = eachslice(X, dims = ndims(X))
    K = [1/(1 + norm(x .- y)) for x in xs, y in xs]
    return det(K)
end
```

Below we generate some random points in $\mathbb{R}^2$ and apply gradient ascent on this function evaluated at the whole array of points. As we can see in the animation below, the points are sent away from each other. In other words, diversity across the array of points increases as we ascend the `ddp_diversity` function.

``` julia
lims = 5
N = 5
X = rand(2,1,N)
T = 50
η = 0.1
anim = @animate for t in 1:T
    X .+= gradient(ddp_diversity, X)[1]
    Z = reshape(X,2,N)
    scatter(
        Z[1,:],Z[2,:],ms=25, 
        xlims=(-lims,lims),ylims=(-lims,lims),
        label="",colour=1:N,
        size=(500,500),
        title="Diverse Counterfactuals"
    )
end
gif(anim, joinpath(www_path, "dice_intro.gif"))
```

![](../../www/dice_intro.gif)

## Usage

The approach can be used in our package as follows:

``` julia
generator = DiCEGenerator()
conv = CounterfactualExplanations.Convergence.GeneratorConditionsConvergence()
ce = generate_counterfactual(
    x, target, counterfactual_data, M, generator; 
    num_counterfactuals=5, convergence=conv
)
plot(ce)
```

![](dice_files/figure-commonmark/cell-5-output-1.svg)

### Effect of Penalty

``` julia
Λ₂ = [0.1, 1.0, 5.0]
ces = []
n_cf = 5
using Flux
for λ₂ ∈ Λ₂  
    λ = [0.00, λ₂]
    generator = DiCEGenerator(λ=λ)
    ces = vcat(
      ces...,
      generate_counterfactual(
            x, target, counterfactual_data, M, generator; 
            num_counterfactuals=n_cf, convergence=conv
      )
    )
end
```

The figure below shows the resulting counterfactual paths. As expected, the resulting counterfactuals are more dispersed across the feature domain for higher choices of $\lambda_2$

![](dice_files/figure-commonmark/cell-7-output-1.svg)

## References

Mothilal, Ramaravind K, Amit Sharma, and Chenhao Tan. 2020. “Explaining Machine Learning Classifiers Through Diverse Counterfactual Explanations.” In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607–17. <https://doi.org/10.1145/3351095.3372850>.

\[1\] With thanks to the respondents on [Discourse](https://discourse.julialang.org/t/getting-around-zygote-mutating-array-issue/83907/2.png)
