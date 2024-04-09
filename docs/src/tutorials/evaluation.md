

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
using CounterfactualExplanations.Objectives: distance
evaluate(ce; measure=distance)
```

    1-element Vector{Vector{Float32}}:
     [3.2273161]

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

    4-element Vector{Vector{Float32}}:
     [2.0]
     [3.2273161]
     [2.7737978]
     [2.7285953]

If no `measure` is specified, the `evaluate` method will return all default measures,

``` julia
evaluate(ce)
```

    3-element Vector{Vector}:
     [1.0]
     Float32[3.2273161]
     [0.0]

which include:

``` julia
CounterfactualExplanations.Evaluation.default_measures
```

    3-element Vector{Function}:
     validity (generic function with 1 method)
     distance (generic function with 1 method)
     redundancy (generic function with 1 method)

### Multiple Measures and Counterfactuals

We can also evaluate multiple counterfactual explanations at once:

``` julia
generator = DiCEGenerator()
ces = generate_counterfactual(x, target, counterfactual_data, M, generator; num_counterfactuals=5)
evaluate(ces)
```

    3-element Vector{Vector}:
     [1.0]
     Float32[3.1955845]
     [[0.0, 0.0, 0.0, 0.0, 0.0]]

By default, each evaluation measure is aggregated across all counterfactual explanations. To return individual measures for each counterfactual explanation you can specify `report_each=true`

``` julia
evaluate(ces; report_each=true)
```

    3-element Vector{Vector}:
     BitVector[[1, 1, 1, 1, 1]]
     Vector{Float32}[[3.3671722, 3.1028512, 3.2829392, 3.0728922, 3.1520686]]
     [[0.0, 0.0, 0.0, 0.0, 0.0]]

## Custom Measures

A `measure` is just a method that takes a `CounterfactualExplanation` as its only positional argument and `agg::Function` as a key argument specifying how measures should be aggregated across counterfactuals. Defining custom measures is therefore straightforward. For example, we could define a measure to compute the inverse target probability as follows:

``` julia
my_measure(ce::CounterfactualExplanation; agg=mean) = agg(1 .- CounterfactualExplanations.target_probs(ce))
evaluate(ce; measure=my_measure)
```

    1-element Vector{Vector{Float32}}:
     [0.41711217]

## Tidy Output

By default, `evaluate` returns vectors of evaluation measures. The optional key argument `output_format::Symbol` can be used to post-process the output in two ways: firstly, to return the output as a dictionary, specify `output_format=:Dict`:

``` julia
evaluate(ces; output_format=:Dict, report_each=true)
```

    Dict{Symbol, Vector} with 3 entries:
      :validity   => BitVector[[1, 1, 1, 1, 1]]
      :redundancy => [[0.0, 0.0, 0.0, 0.0, 0.0]]
      :distance   => Vector{Float32}[[3.36717, 3.10285, 3.28294, 3.07289, 3.15207]]

Secondly, to return the output as a data frame, specify `output_format=:DataFrame`.

``` julia
evaluate(ces; output_format=:DataFrame, report_each=true)
```

By default, data frames are pivoted to long format using individual counterfactuals as the `id` column. This behaviour can be suppressed by specifying `pivot_longer=false`.

## Multiple Counterfactual Explanations

It may be necessary to generate counterfactual explanations for multiple individuals.

Below, for example, we first select multiple samples (5) from the non-target class and then generate counterfactual explanations for all of them.

This can be done using broadcasting:

``` julia
# Factual and target:
ids = rand(findall(predict_label(M, counterfactual_data) .== factual), n_individuals)
xs = select_factual(counterfactual_data, ids)
ces = generate_counterfactual(xs, target, counterfactual_data, M, generator; num_counterfactuals=5)
evaluation = evaluate.(ces)
```

    5-element Vector{Vector{Vector}}:
     [[1.0], Float32[3.351181], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
     [[1.0], Float32[2.6405892], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
     [[1.0], Float32[2.935012], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
     [[1.0], Float32[3.5348382], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
     [[1.0], Float32[3.9373996], [[0.0, 0.0, 0.0, 0.0, 0.0]]]

    Vector{Vector}[[[1.0], Float32[3.351181], [[0.0, 0.0, 0.0, 0.0, 0.0]]], [[1.0], Float32[2.6405892], [[0.0, 0.0, 0.0, 0.0, 0.0]]], [[1.0], Float32[2.935012], [[0.0, 0.0, 0.0, 0.0, 0.0]]], [[1.0], Float32[3.5348382], [[0.0, 0.0, 0.0, 0.0, 0.0]]], [[1.0], Float32[3.9373996], [[0.0, 0.0, 0.0, 0.0, 0.0]]]]

This leads us to our next topic: Performance Benchmarks.
