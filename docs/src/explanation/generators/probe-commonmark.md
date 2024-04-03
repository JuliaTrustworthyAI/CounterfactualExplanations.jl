

# `ProbeGenerator`

``` @meta›
CurrentModule = CounterfactualExplanations 
```

The `ProbeGenerator` is designed to navigate the trade-offs between costs and robustness in Algorithmic Recourse (Pawelczyk et al. 2022).

## Description

The goal of ProbeGenerator is to find a recourse x’ whose prediction at any point y within some set around x’ belongs to the positive class with probability 1 - r, where r is the recourse invalidation rate. It minimizes the gap between the achieved and desired recourse invalidation rates, minimizes recourse costs, and also ensures that the resulting recourse achieves a positive model prediction.

### Explanation

The loss function this generator is defined below. R is a hinge loss parameter which helps control for robustness. The loss and penalty functions can still be chosen freely.

``` math
\begin{aligned}
R(x'; \sigma^2 I) + l(f(x'), s) + \lambda d_c(x', x)
\end{aligned}
```

R uses the following formula to control for noise. It generates small perturbations and checks how often the counterfactual explanation flips back to a factual one, when small amounts of noise are added to it.

``` math
\begin{aligned}
\Delta(x^{\hat{E}}) &= E_{\varepsilon}[h(x^{\hat{E}}) - h(x^{\hat{E}} + \varepsilon)]
\end{aligned}
```

The above formula is not differentiable. For this reason the generator uses the closed form version of the formula below.

``` math
\begin{equation}
\Delta \tilde{}(x^{\hat{E}}, \sigma^2 I) = 1 - \Phi \left(\frac{\sqrt{f(x^{\hat{E}})}}{\sqrt{\nabla f(x^{\hat{E}})^T \sigma^2 I \nabla f(x^{\hat{E}})}}\right) 
\end{equation}
```

### Usage

Generating a counterfactual with the data loaded and generator chosen works as follows:

Note: It is important to set the convergence to “:invalidation_rate” here.

``` julia
M = fit_model(counterfactual_data, :DeepEnsemble)
opt = Descent(0.01)
generator = CounterfactualExplanations.Generators.ProbeGenerator(opt=opt)
ce = generate_counterfactual(x, target, counterfactual_data, M, generator, converge_when =:invalidation_rate, invalidation_rate = 0.5, learning_rate = 0.5)
plot(ce)
```

Choosing different invalidation rates makes the counterfactual more or less robust. The following plot shows the counterfactuals generated for different invalidation rates.

![](probe_files/figure-commonmark/cell-4-output-1.svg)

## References

Pawelczyk, Martin, Teresa Datta, Johannes van-den-Heuvel, Gjergji Kasneci, and Himabindu Lakkaraju. 2022. “Probabilistically Robust Recourse: Navigating the Trade-Offs Between Costs and Robustness in Algorithmic Recourse.” *arXiv Preprint arXiv:2203.06768*.
