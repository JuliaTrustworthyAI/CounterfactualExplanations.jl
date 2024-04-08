

``` @meta
CurrentModule = CounterfactualExplanations 
```

# [Counterfactual Generators](@id generators_explanation)

Counterfactual generators form the very core of this package. The [`generator_catalogue`](@ref) can be used to inspect the available generators:

``` julia
generator_catalogue
```

    Dict{Symbol, Any} with 11 entries:
      :gravitational   => GravitationalGenerator
      :growing_spheres => GrowingSpheresGenerator
      :revise          => REVISEGenerator
      :clue            => CLUEGenerator
      :probe           => ProbeGenerator
      :dice            => DiCEGenerator
      :feature_tweak   => FeatureTweakGenerator
      :claproar        => ClaPROARGenerator
      :wachter         => WachterGenerator
      :generic         => GenericGenerator
      :greedy          => GreedyGenerator

The following sections provide brief descriptions of all of them.

## Gradient-based Counterfactual Generators

At the time of writing, all generators are gradient-based: that is, counterfactuals are searched through gradient descent. In Altmeyer et al. (2023) we lay out a general methodological framework that can be applied to all of these generators:

``` math
\begin{aligned}
\mathbf{s}^\prime &= \arg \min_{\mathbf{s}^\prime \in \mathcal{S}} \left\{  {\text{yloss}(M(f(\mathbf{s}^\prime)),y^*)}+ \lambda {\text{cost}(f(\mathbf{s}^\prime)) }  \right\} 
\end{aligned} 
```

“Here $\mathbf{s}^\prime=\left\{s_k^\prime\right\}_K$ is a $K$-dimensional array of counterfactual states and $f: \mathcal{S} \mapsto \mathcal{X}$ maps from the counterfactual state space to the feature space.” (Altmeyer et al. 2023)

For most generators, the state space *is* the feature space ($f$ is the identity function) and the number of counterfactuals $K$ is one. Latent Space generators instead search counterfactuals in some latent space $\mathcal{S}$. In this case, $f$ corresponds to the decoder part of the generative model, that is the function that maps back from the latent space to inputs.

## References

Altmeyer, Patrick, Giovan Angela, Aleksander Buszydlik, Karol Dobiczek, Arie van Deursen, and Cynthia Liem. 2023. “Endogenous Macrodynamics in Algorithmic Recourse.” In *First IEEE Conference on Secure and Trustworthy Machine Learning*. <https://doi.org/10.1109/satml54575.2023.00036>.
