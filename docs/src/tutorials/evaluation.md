
``` @meta
CurrentModule = CounterfactualExplanations 
```

# Performance Evaluation

Now that we know how to generate counterfactual explanations in Julia, you may have a few follow-up questions: How do I know if the counterfactual search has been successful? How good is my counterfactual explanation? What does ‘good’ even mean in this context? In this tutorial, we will see how counterfactual explanations can be evaluated with respect to their performance.

## Default Measures

Numerous evaluation measures for counterfactual explanations have been proposed. In what follows, we will cover some of the most important measures.

### Single Measure, Single Counterfactual

One of the most important measures is [`validity`](@ref), which simply determines whether or not a counterfactual explanation $x^{\prime}$ is valid in the sense that it yields the target prediction: $M(x^{\prime})=t$. We can evaluate the validity of a single counterfactual explanation `ce` using the [`Evaluation.evaluate`](@ref) function as follows:

``` julia
using CounterfactualExplanations.Evaluation: evaluate, validity
evaluate(ce; measure=validity)
```

    1-element Vector{Vector{Float64}}:
     [1.0]

For a single counterfactual explanation, this evaluation measure can only take two values: it is either equal to `1`, if the explanation is valid or `0` otherwise. Another important measure is [`distance`](@ref), which relates to the distance between the factual $x$ and the counterfactual $x^{\prime}$. In the context of Algorithmic Recourse, higher distances are typically associated with higher costs to individuals seeking recourse.

``` julia
using CounterfactualExplanations.Evaluation: distance
evaluate(ce; measure=distance)
```

    1-element Vector{Vector{Float64}}:
     [0.7746574005639171]

By default, `distance` computes the L2 (Euclidean) distance.

### Multiple Measures, Single Counterfactual

You might be interested in computing not just the L2 distance, but various LP norms. This can be done by supplying a vector of functions to the `measure` key argument. For convenience, all default distance measures have already been collected in a vector:

``` julia
using CounterfactualExplanations.Evaluation: distance_measures
distance_measures
```

    4-element Vector{Function}:
     distance_l0 (generic function with 1 method)
     distance_l1 (generic function with 1 method)
     distance_l2 (generic function with 1 method)
     distance_linf (generic function with 1 method)

We can use this vector of evaluation measures as follows:

``` julia
evaluate(ce; measure=distance_measures)
```

    4-element Vector{Vector{Float64}}:
     [2.0]
     [1.0941725863975873]
     [0.7746574005639171]
     [0.5743559084917548]

If no `measure` is specified, the `evaluate` method will return all default measures,

``` julia
evaluate(ce)
```

    3-element Vector{Vector{Float64}}:
     [1.0]
     [0.7746574005639171]
     [0.0]

which include:

``` julia
CounterfactualExplanations.Evaluation.default_measures
```

    3-element Vector{Function}:
     validity (generic function with 1 method)
     distance (generic function with 2 methods)
     redundancy (generic function with 1 method)

### Multiple Measures and Counterfactuals

We can also evaluate multiple counterfactual explanations at once:

``` julia
generator = DiCEGenerator()
ces = generate_counterfactual(x, target, counterfactual_data, M, generator; num_counterfactuals=5)
evaluate(ces)
```

    3-element Vector{Vector{Float64}}:
     [1.0]
     [1.2301144669473314]
     [0.0]

By default, each evaluation measure is aggregated across all counterfactual explanations. To return individual measures for each counterfactual explanation you can specify `report_each=true`

``` julia
evaluate(ces; report_each=true)
```

    3-element Vector{AbstractVector}:
     Bool[1, 1, 1, 1, 1]
     [1.2569921340312415, 1.2264824140495325, 1.1942210044404975, 1.2486051151163093, 1.224271667099076]
     [0.0, 0.0, 0.0, 0.0, 0.0]

## Custom Measures

A `measure` is just a method that takes a `CounterfactualExplanation` as its only positional argument. Defining custom measures is therefore straightforward. For example, we could define a measure to compute the inverse target probability as follows:

``` julia
my_measure(ce::CounterfactualExplanation) = 1 .- CounterfactualExplanations.target_probs(ce)
evaluate(ce; measure=my_measure)
```

    1-element Vector{Vector{Float64}}:
     [0.3637938396750189]

## Tidy Output

By default, `evaluate` returns vectors of evaluation measures. The optional key argument `output_format::Symbol` can be used to post-process the output in two ways: firstly, to return the output as a dictionary, specify `output_format=:Dict`:

``` julia
evaluate(ces; output_format=:Dict, report_each=true)
```

    Dict{Symbol, AbstractVector} with 3 entries:
      :validity   => Bool[1, 1, 1, 1, 1]
      :redundancy => [0.0, 0.0, 0.0, 0.0, 0.0]
      :distance   => [1.25699, 1.22648, 1.19422, 1.24861, 1.22427]

Secondly, to return the output as a data frame, specify `output_format=:DataFrame`.

``` julia
evaluate(ces; output_format=:DataFrame, report_each=true)
```

By default, data frames are pivoted to long format using individual counterfactuals as the `id` column. This behaviour can be suppressed by specifying `pivot_longer=false`.
