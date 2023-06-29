# Performance Evaluation

``` @meta
CurrentModule = CounterfactualExplanations 
```

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
     [3.6304846]

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
     [3.6304846]
     [2.9974036]
     [2.9093676]

If no `measure` is specified, the `evaluate` method will return all default measures,

``` julia
evaluate(ce)
```

    3-element Vector{Vector}:
     [1.0]
     Float32[3.6304846]
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
     Float32[3.5359674]
     [[0.0, 0.0, 0.0, 0.0, 0.0]]

By default, each evaluation measure is aggregated across all counterfactual explanations. To return individual measures for each counterfactual explanation you can specify `report_each=true`

``` julia
evaluate(ces; report_each=true)
```

    3-element Vector{Vector}:
     BitVector[[1, 1, 1, 1, 1]]
     Vector{Float32}[[3.8770418, 3.3639247, 3.653852, 3.310768, 3.4742484]]
     [[0.0, 0.0, 0.0, 0.0, 0.0]]

## Custom Measures

A `measure` is just a method that takes a `CounterfactualExplanation` as its only positional argument and `agg::Function` as a key argument specifying how measures should be aggregated across counterfactuals. Defining custom measures is therefore straightforward. For example, we could define a measure to compute the inverse target probability as follows:

``` julia
my_measure(ce::CounterfactualExplanation; agg=mean) = agg(1 .- CounterfactualExplanations.target_probs(ce))
evaluate(ce; measure=my_measure)
```

    1-element Vector{Vector{Float32}}:
     [0.22146827]

## Tidy Output

By default, `evaluate` returns vectors of evaluation measures. The optional key argument `output_format::Symbol` can be used to post-process the output in two ways: firstly, to return the output as a dictionary, specify `output_format=:Dict`:

``` julia
evaluate(ces; output_format=:Dict, report_each=true)
```

    Dict{Symbol, Vector} with 3 entries:
      :validity   => BitVector[[1, 1, 1, 1, 1]]
      :redundancy => [[0.0, 0.0, 0.0, 0.0, 0.0]]
      :distance   => Vector{Float32}[[3.87704, 3.36392, 3.65385, 3.31077, 3.47425]]

Secondly, to return the output as a data frame, specify `output_format=:DataFrame`.

``` julia
evaluate(ces; output_format=:DataFrame, report_each=true)
```

By default, data frames are pivoted to long format using individual counterfactuals as the `id` column. This behaviour can be suppressed by specifying `pivot_longer=false`.

## Multiple Counterfactual Explanations

It may be necessary to generate counterfactual explanations for multiple individuals.

Below, for example, we first select multiple samples (5) from the non-target class and then generate counterfactual explanations for all of them.

``` julia
# Factual and target:
ids = rand(findall(predict_label(M, counterfactual_data) .== factual), n_individuals)
xs = select_factual(counterfactual_data, ids)
ces = generate_counterfactual(xs, target, counterfactual_data, M, generator; num_counterfactuals=5)
evaluation = evaluate(ces)
```

    15×4 DataFrame
     Row │ sample  num_counterfactual  variable    value                     
         │ Int64   Int64               String      Any                       
    ─────┼───────────────────────────────────────────────────────────────────
       1 │      1                   1  distance    3.51446
       2 │      1                   1  redundancy  [0.0, 0.0, 0.0, 0.0, 0.0]
       3 │      1                   1  validity    1.0
       4 │      2                   1  distance    3.15563
       5 │      2                   1  redundancy  [0.0, 0.0, 0.0, 0.0, 0.0]
       6 │      2                   1  validity    1.0
       7 │      3                   1  distance    3.49398
       8 │      3                   1  redundancy  [0.0, 0.0, 0.0, 0.0, 0.0]
       9 │      3                   1  validity    1.0
      10 │      4                   1  distance    4.58058
      11 │      4                   1  redundancy  [0.0, 0.0, 0.0, 0.0, 0.0]
      12 │      4                   1  validity    0.8
      13 │      5                   1  distance    3.66033
      14 │      5                   1  redundancy  [0.0, 0.0, 0.0, 0.0, 0.0]
      15 │      5                   1  validity    0.8

This leads us to our next topic: Performance Benchmarks.
