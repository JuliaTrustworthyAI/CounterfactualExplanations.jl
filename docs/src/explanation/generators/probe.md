``` @meta
CurrentModule = CounterfactualExplanations 
```

# `ProbeGenerator`

The `ProbeGenerator` is designed to navigate the trade-offs between costs and robustness in algorithmic recourse.

## Description

The goal of ProbeGenerator is to find a recourse x' whose prediction at any point y within some set around x' belongs to the positive class with probability 1 - r, where r is the recourse invalidation rate. It minimizes the gap between the achieved and desired recourse invalidation rates, minimizes recourse costs, and also ensures that the resulting recourse achieves a positive model prediction.
Usage

### Explanation

``` math
\begin{aligned}
\Delta \tilde{}(x^{\hat{E}}, \sigma^2 I) &= 1 - \Phi \left(\frac{\sqrt{f(x^{\hat{E}})}}{\sqrt{\nabla f(x^{\hat{E}})^T \sigma^2 I \nabla f(x^{\hat{E}})}}\right) \tag{4}
\end{aligned}
```

``` math
\begin{aligned}
R(x'; \sigma^2 I) + l(f(x'), s) + \lambda d_c(x', x)
\end{aligned}
```

``` math
\begin{aligned}
\Delta(x^{\hat{E}}) &= E_{\varepsilon}[h(x^{\hat{E}}) - h(x^{\hat{E}} + \varepsilon)]
\end{aligned}
```


### Usage

```julia
using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Generators

counterfactual_data = load_linearly_separable()
M = fit_model(counterfactual_data, :Linear)
target = 2
factual = 1
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)
generator = ProbeGenerator()
```


Then you can use the generator to produce a counterfactual as follows:

```julia
linear_counterfactual = generate_counterfactual(
    x,
    target,
    counterfactual_data,
    M,
    generator;
    converge_when=:invalidation_rate,
    max_iter=1000,
    invalidation_rate=0.1,
    learning_rate=0.1,
)

CounterfactualExplanations.plot(linear_counterfactual)
```


## References

Our Framework: Probabilistically Robust Recourse. "Probabilistically Robust Recourse: Navigating the Trade-offs between Costs and Robustness in Algorithmic Recourse". University of Tübingen, 2022.

