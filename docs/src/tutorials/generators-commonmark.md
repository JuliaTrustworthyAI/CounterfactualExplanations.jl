

``` @meta
CurrentModule = CounterfactualExplanations 
```

# Handling Generators

Generating Counterfactual Explanations can be seen as a generative modelling task because it involves generating samples in the input space: $x \sim \mathcal{X}$. In this tutorial, we will introduce how Counterfactual `GradientBasedGenerator`s are used. They are discussed in more detail in the explanatory section of the documentation.

## Composable Generators

!!! warning “Breaking Changes Expected”
Work on this feature is still in its very early stages and breaking changes should be expected.

One of the key objectives for this package is **Composability**. It turns out that many of the various counterfactual generators that have been proposed in the literature, essentially do the same thing: they optimize an objective function. Formally we have,

$$
\begin{aligned}
\mathbf{s}^\prime &= \arg \min_{\mathbf{s}^\prime \in \mathcal{S}} \left\{  {\text{yloss}(M(f(\mathbf{s}^\prime)),y^*)}+ \lambda {\text{cost}(f(\mathbf{s}^\prime)) }  \right\} 
\end{aligned} 
 \qquad(1)$$

where $\text{yloss}$ denotes the main loss function and $\text{cost}$ is a penalty term (Altmeyer et al. 2023).

Without going into further detail here, the important thing to mention is that [Equation 1](#eq-general) very closely describes how counterfactual search is actually implemented in the package. In other words, all off-the-shelf generators currently implemented work with that same objective. They just vary in the way that penalties are defined, for example. This gives rise to an interesting idea:

> Why not compose generators that combine ideas from different off-the-shelf generators?

The [`GradientBasedGenerator`](@ref) class provides a straightforward way to do this, without requiring users to build custom `GradientBasedGenerator`s from scratch. It can be instantiated as follows:

``` julia
generator = GradientBasedGenerator()
```

By default, this creates a `generator` that simply performs gradient descent without any penalties. To modify the behaviour of the `generator`, you can define the counterfactual search objective function using the [`@objective`](@ref) macro:

``` julia
@objective(generator, logitbinarycrossentropy + 0.1distance_l2 + 1.0ddp_diversity)
```

Here we have essentially created a version of the [`DiCEGenerator`](@ref):

``` julia
ce = generate_counterfactual(x, target, counterfactual_data, M, generator; num_counterfactuals=5)
plot(ce)
```

![](generators_files/figure-commonmark/cell-5-output-1.svg)

Multiple macros can be chained using `Chains.jl` making it easy to create entirely new flavours of counterfactual generators. The following generator, for example, combines ideas from DiCE (Mothilal, Sharma, and Tan 2020) and REVISE (Joshi et al. 2019):

``` julia
@chain generator begin
    @objective logitcrossentropy + 1.0ddp_diversity     # DiCE (Mothilal et al. 2020)
    @with_optimiser Flux.Adam(0.1)                      
    @search_latent_space                                # REVISE (Joshi et al. 2019)
end
```

Let’s take this generator to our MNIST dataset and generate a counterfactual explanation for turning a 0 into a 8.

![](generators_files/figure-commonmark/cell-10-output-1.svg)

## Off-the-Shelf Generators

Off-the-shelf generators are just default recipes for counterfactual generators. Currently, the following off-the-shelf counterfactual generators are implemented in the package:

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

To specify the type of generator you want to use, you can simply instantiate it:

``` julia
# Search:
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
plot(ce)
```

![](generators_files/figure-commonmark/cell-13-output-1.svg)

We generally make an effort to follow the literature as closely as possible when implementing off-the-shelf generators.

## References

Altmeyer, Patrick, Giovan Angela, Aleksander Buszydlik, Karol Dobiczek, Arie van Deursen, and Cynthia Liem. 2023. “Endogenous Macrodynamics in Algorithmic Recourse.” In *First IEEE Conference on Secure and Trustworthy Machine Learning*. <https://doi.org/10.1109/satml54575.2023.00036>.

Joshi, Shalmali, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh. 2019. “Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems.” <https://arxiv.org/abs/1907.09615>.

Mothilal, Ramaravind K, Amit Sharma, and Chenhao Tan. 2020. “Explaining Machine Learning Classifiers Through Diverse Counterfactual Explanations.” In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607–17. <https://doi.org/10.1145/3351095.3372850>.
