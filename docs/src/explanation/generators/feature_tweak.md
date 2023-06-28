# `FeatureTweakGenerator`

``` @meta
CurrentModule = CounterfactualExplanations 
```

**Feature Tweak** refers to the generator introduced by Tolomei et al. (2017). Our implementation takes inspiration from the [featureTweakPy library](https://github.com/upura/featureTweakPy).

## Description

Feature Tweak is a powerful recourse algorithm for ensembles of tree-based classifiers such as random forests. Though the problem of understanding how an input to an ensemble model could be transformed in such a way that the model changes its original prediction has been proven to be NP-hard \[1\], Feature Tweak provides an algorithm that manages to tractably solve this problem in multiple real-world applications. An example of a problem Feature Tweak is able to efficiently solve, explored in depth in \[1\], is the problem of transforming an advertisement that has been classified by the ensemble model as a low-quality advertisement to a high-quality one through small changes to its features. With the help of Feature Tweak, advertisers can both learn about the reasons a particular ad was marked to have a low quality, as well as receive actionable suggestions about how to convert a low-quality ad into a high-quality one.

Though Feature Tweak is a powerful way of avoiding brute-force search in an exponential search space, it does not come without disadvantages. The primary limitations of the approach are that it’s currently only applicable to tree-based classifiers and works only in the setting of binary classification. Another problem is that though the algorithm avoids exponential-time search, it is often still computationally expensive. The algorithm may be improved in the future to tackle all of these shortcomings.

The following equation displays how a true negative instance x can be transformed into a positively predicted instance **x’**. To be more precise, **x’** is the best possible transformation among all transformations **x\***, computed with a cost function δ.

``` math
\begin{aligned}
\mathbf{x}^\prime = \arg_{\mathbf{x^*}} \min \{ {\delta(\mathbf{x}, \mathbf{x^*}) | \hat{f}(\mathbf{x}) = -1 \wedge \hat{f}(\mathbf{x^*}) = +1} \}
\end{aligned}
```

## References

Tolomei, Gabriele, Fabrizio Silvestri, Andrew Haines, and Mounia Lalmas. 2017. “Interpretable Predictions of Tree-Based Ensembles via Actionable Feature Tweaking.” In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 465–74. <https://doi.org/10.1145/3097983.3098039>.
