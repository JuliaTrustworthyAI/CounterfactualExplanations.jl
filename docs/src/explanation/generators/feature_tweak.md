
``` @meta
CurrentModule = CounterfactualExplanations 
```

# `FeatureTweakGenerator`

**Feature Tweaking** refers to the generator introduced by Tolomei et al. (2017)

## Description

The following equation displays how a true negative instance x can be transformed to a positively predicted instance x'. To be more precise, x' is the best possible transformation among all transformations x*, computed with a cost function \delta.

```math
\begin{aligned}
\mathbf{x}^\prime &= \arg_{\mathbf{x^*}} \min \{ {\delta(\mathbf{x}, \mathbf{x^*}) | \hat{f}(\mathbf{x}) &= -1 \wedge \hat{f}(\mathbf{x^*}) &= +1} \}
\end{aligned}
```

## References

Tolomei, Silvestri, Haines, and Lalmas. 2017. "Interpretable Predictions of Tree-based Ensembles via Actionable Feature Tweaking" https://arxiv.org/abs/1706.06691
