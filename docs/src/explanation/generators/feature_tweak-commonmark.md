

``` @meta
CurrentModule = CounterfactualExplanations 
```

# `FeatureTweakGenerator`

**Feature Tweak** refers to the generator introduced by Tolomei et al. (2017). Our implementation takes inspiration from the [featureTweakPy library](https://github.com/upura/featureTweakPy).

## Description

Feature Tweak is a powerful recourse algorithm for ensembles of tree-based classifiers such as random forests. Though the problem of understanding how an input to an ensemble model could be transformed in such a way that the model changes its original prediction has been proven to be NP-hard (Tolomei et al. 2017), Feature Tweak provides an algorithm that manages to tractably solve this problem in multiple real-world applications. An example of a problem Feature Tweak is able to efficiently solve, explored in depth in Tolomei et al. (2017) is the problem of transforming an advertisement that has been classified by the ensemble model as a low-quality advertisement to a high-quality one through small changes to its features. With the help of Feature Tweak, advertisers can both learn about the reasons a particular ad was marked to have a low quality, as well as receive actionable suggestions about how to convert a low-quality ad into a high-quality one.

Though Feature Tweak is a powerful way of avoiding brute-force search in an exponential search space, it does not come without disadvantages. The primary limitations of the approach are that it’s currently only applicable to tree-based classifiers and works only in the setting of binary classification. Another problem is that though the algorithm avoids exponential-time search, it is often still computationally expensive. The algorithm may be improved in the future to tackle all of these shortcomings.

The following equation displays how a true negative instance x can be transformed into a positively predicted instance **x’**. To be more precise, **x’** is the best possible transformation among all transformations **x\***, computed with a cost function δ.

``` math
\begin{aligned}
\mathbf{x}^\prime = \arg_{\mathbf{x^*}} \min \{ {\delta(\mathbf{x}, \mathbf{x^*}) | \hat{f}(\mathbf{x}) = -1 \wedge \hat{f}(\mathbf{x^*}) = +1} \}
\end{aligned}
```

## Example

In this example we apply the Feature Tweak algorithm to a decision tree and a random forest trained on the moons dataset. We first load the data and fit the models:

``` julia
n = 500
counterfactual_data = CounterfactualData(TaijaData.load_moons(n)...)

# Classifiers
decision_tree = CounterfactualExplanations.Models.fit_model(
    counterfactual_data, :DecisionTree; max_depth=5, min_samples_leaf=3
)
forest = Models.fit_model(counterfactual_data, :RandomForest)
```

Next, we select a point to explain and a target class to transform the point to. We then search for counterfactuals using the `FeatureTweakGenerator`:

``` julia
# Select a point to explain:
x = float32.([1, -0.5])[:,:]
factual = Models.predict_label(forest, x)
target = counterfactual_data.y_levels[findall(counterfactual_data.y_levels != factual)][1]

# Search for counterfactuals:
generator = FeatureTweakGenerator(ϵ=0.1)
tree_counterfactual = generate_counterfactual(
    x, target, counterfactual_data, decision_tree, generator
)
forest_counterfactual = generate_counterfactual(
    x, target, counterfactual_data, forest, generator
)
```

The resulting counterfactuals are shown below:

``` julia
p1 = plot(
    tree_counterfactual;
    colorbar=false,
    title="Decision Tree",
)

p2 = plot(
    forest_counterfactual; title="Random Forest",
    colorbar=false,
)

display(plot(p1, p2; size=(800, 400)))
```

![](feature_tweak_files/figure-commonmark/cell-5-output-1.svg)

## References

Tolomei, Gabriele, Fabrizio Silvestri, Andrew Haines, and Mounia Lalmas. 2017. “Interpretable Predictions of Tree-Based Ensembles via Actionable Feature Tweaking.” In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 465–74. <https://doi.org/10.1145/3097983.3098039>.
