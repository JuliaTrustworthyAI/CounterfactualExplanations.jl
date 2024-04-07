

``` @meta
CurrentModule = CounterfactualExplanations 
```

# Performance Benchmarks

In the previous tutorial, we have seen how counterfactual explanations can be evaluated. An important follow-up task is to compare the performance of different counterfactual generators is an important task. Researchers can use benchmarks to test new ideas they want to implement. Practitioners can find the right counterfactual generator for their specific use case through benchmarks. In this tutorial, we will see how to run benchmarks for counterfactual generators.

## Post Hoc Benchmarking

We begin by continuing the discussion from the previous tutorial: suppose you have generated multiple counterfactual explanations for multiple individuals, like below:

``` julia
# Factual and target:
n_individuals = 5
ids = rand(findall(predict_label(M, counterfactual_data) .== factual), n_individuals)
xs = select_factual(counterfactual_data, ids)
ces = generate_counterfactual(xs, target, counterfactual_data, M, generator; num_counterfactuals=5)
```

You may be interested in comparing the outcomes across individuals. To benchmark the various counterfactual explanations using default evaluation measures, you can simply proceed as follows:

``` julia
bmk = benchmark(ces)
```

Under the hood, the [`benchmark(counterfactual_explanations::Vector{CounterfactualExplanation})`](@ref) uses [`evaluate(counterfactual_explanations::Vector{CounterfactualExplanation})`](@ref) to generate a [`Benchmark`](@ref) object, which contains the evaluation in its most granular form as a `DataFrame`.

### Working with `Benchmark`s

For convenience, the `DataFrame` containing the evaluation can be returned by simply calling the `Benchmark` object. By default, the aggregated evaluation measures across `id` (in line with the default behaviour of `evaluate`).

``` julia
bmk()
```

    15×7 DataFrame
     Row │ sample                                variable    value    generator    ⋯
         │ Base.UUID                             String      Float64  Symbol       ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03  distance    3.17243  GradientBase ⋯
       2 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03  redundancy  0.0      GradientBase
       3 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03  validity    1.0      GradientBase
       4 │ 1cfad5e6-f42e-11ee-301b-ed0f5ad5da41  distance    3.07148  GradientBase
       5 │ 1cfad5e6-f42e-11ee-301b-ed0f5ad5da41  redundancy  0.0      GradientBase ⋯
       6 │ 1cfad5e6-f42e-11ee-301b-ed0f5ad5da41  validity    1.0      GradientBase
       7 │ 1cfadb9a-f42e-11ee-3f4d-e38b20922cc3  distance    3.62159  GradientBase
       8 │ 1cfadb9a-f42e-11ee-3f4d-e38b20922cc3  redundancy  0.0      GradientBase
       9 │ 1cfadb9a-f42e-11ee-3f4d-e38b20922cc3  validity    1.0      GradientBase ⋯
      10 │ 1cfadfe8-f42e-11ee-116e-8da8b6d04de4  distance    2.62783  GradientBase
      11 │ 1cfadfe8-f42e-11ee-116e-8da8b6d04de4  redundancy  0.0      GradientBase
      12 │ 1cfadfe8-f42e-11ee-116e-8da8b6d04de4  validity    1.0      GradientBase
      13 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec  distance    2.91985  GradientBase ⋯
      14 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec  redundancy  0.0      GradientBase
      15 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec  validity    1.0      GradientBase
                                                                   4 columns omitted

To retrieve the granular dataset, simply do:

``` julia
bmk(agg=nothing)
```

    75×8 DataFrame
     Row │ sample                                num_counterfactual  variable    v ⋯
         │ Base.UUID                             Int64               String      F ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   1  distance    3 ⋯
       2 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   2  distance    3
       3 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   3  distance    3
       4 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   4  distance    3
       5 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   5  distance    3 ⋯
       6 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   1  redundancy  0
       7 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   2  redundancy  0
       8 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   3  redundancy  0
       9 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   4  redundancy  0 ⋯
      10 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   5  redundancy  0
      11 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03                   1  validity    1
      ⋮  │                  ⋮                            ⋮               ⋮         ⋱
      66 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   1  redundancy  0
      67 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   2  redundancy  0 ⋯
      68 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   3  redundancy  0
      69 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   4  redundancy  0
      70 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   5  redundancy  0
      71 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   1  validity    1 ⋯
      72 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   2  validity    1
      73 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   3  validity    1
      74 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   4  validity    1
      75 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec                   5  validity    1 ⋯
                                                       5 columns and 54 rows omitted

Since benchmarks return a `DataFrame` object on call, post-processing is straightforward. For example, we could use [`Tidier.jl`](https://kdpsingh.github.io/Tidier.jl/dev/):

``` julia
using Tidier
@chain bmk() begin
    @filter(variable == "distance")
    @select(sample, variable, value)
end
```

    [ Info: Precompiling Tidier [f0413319-3358-4bb0-8e7c-0c83523a93bd]
    [ Info: Precompiling GeometryBasicsExt [b238bd29-021f-5edc-8b0e-16b9cda5f63a]

    5×3 DataFrame
     Row │ sample                                variable  value   
         │ Base.UUID                             String    Float64 
    ─────┼─────────────────────────────────────────────────────────
       1 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03  distance  3.17243
       2 │ 1cfad5e6-f42e-11ee-301b-ed0f5ad5da41  distance  3.07148
       3 │ 1cfadb9a-f42e-11ee-3f4d-e38b20922cc3  distance  3.62159
       4 │ 1cfadfe8-f42e-11ee-116e-8da8b6d04de4  distance  2.62783
       5 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec  distance  2.91985

### Metadata for Counterfactual Explanations

Benchmarks always report metadata for each counterfactual explanation, which is automatically inferred by default. The default metadata concerns the explained `model` and the employed `generator`. In the current example, we used the same model and generator for each individual:

``` julia
@chain bmk() begin
    @group_by(sample)
    @select(sample, model, generator)
    @summarize(model=first(model),generator=first(generator))
    @ungroup
end
```

    5×3 DataFrame
     Row │ sample                                model                             ⋯
         │ Base.UUID                             Symbol                            ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ 1cf32774-f42e-11ee-1dc3-c14a27548b03  FluxModel(Chain(Dense(2 => 2)), … ⋯
       2 │ 1cfad5e6-f42e-11ee-301b-ed0f5ad5da41  FluxModel(Chain(Dense(2 => 2)), …
       3 │ 1cfadb9a-f42e-11ee-3f4d-e38b20922cc3  FluxModel(Chain(Dense(2 => 2)), …
       4 │ 1cfadfe8-f42e-11ee-116e-8da8b6d04de4  FluxModel(Chain(Dense(2 => 2)), …
       5 │ 1cfae3f6-f42e-11ee-0e0b-dfd18963acec  FluxModel(Chain(Dense(2 => 2)), … ⋯
                                                                    1 column omitted

Metadata can also be provided as an optional key argument.

``` julia
meta_data = Dict(
    :generator => "Generic",
    :model => "MLP",
)
meta_data = [meta_data for i in 1:length(ces)]
bmk = benchmark(ces; meta_data=meta_data)
@chain bmk() begin
    @group_by(sample)
    @select(sample, model, generator)
    @summarize(model=first(model),generator=first(generator))
    @ungroup
end
```

    5×3 DataFrame
     Row │ sample                                model   generator 
         │ Base.UUID                             String  String    
    ─────┼─────────────────────────────────────────────────────────
       1 │ efe3f866-f42e-11ee-2eed-6b8741e13460  MLP     Generic
       2 │ efe6c708-f42e-11ee-2914-0136f102d9af  MLP     Generic
       3 │ efe6cadc-f42e-11ee-2f0e-df68cf455760  MLP     Generic
       4 │ efe6cdf2-f42e-11ee-3940-9138efb62d55  MLP     Generic
       5 │ efe6d0ae-f42e-11ee-366f-4bf618ab1e59  MLP     Generic

## Ad Hoc Benchmarking

So far we have assumed the following workflow:

1.  Fit some machine learning model.
2.  Generate counterfactual explanations for some individual(s) (`generate_counterfactual`).
3.  Evaluate and benchmark them (`benchmark(ces::Vector{CounterfactualExplanation})`).

In many cases, it may be preferable to combine these steps. To this end, we have added support for two scenarios of Ad Hoc Benchmarking.

### Pre-trained Models

In the first scenario, it is assumed that the machine learning models have been pre-trained and so the workflow can be summarized as follows:

1.  Fit some machine learning model(s).
2.  Generate counterfactual explanations and benchmark them.

We suspect that this is the most common workflow for practitioners who are interested in benchmarking counterfactual explanations for the pre-trained machine learning models. Let’s go through this workflow using a simple example. We first train some models and store them in a dictionary:

``` julia
models = Dict(
    :MLP => fit_model(counterfactual_data, :MLP),
    :Linear => fit_model(counterfactual_data, :Linear),
)
```

Next, we store the counterfactual generators of interest in a dictionary as well:

``` julia
generators = Dict(
    :Generic => GenericGenerator(),
    :Gravitational => GravitationalGenerator(),
    :Wachter => WachterGenerator(),
    :ClaPROAR => ClaPROARGenerator(),
)
```

Then we can run a benchmark for individual(s) `x`, a pre-specified `target` and `counterfactual_data` as follows:

``` julia
bmk = benchmark(x, target, counterfactual_data; models=models, generators=generators)
```

In this case, metadata is automatically inferred from the dictionaries:

``` julia
@chain bmk() begin
    @filter(variable == "distance")
    @select(sample, variable, value, model, generator)
end
```

    8×5 DataFrame
     Row │ sample                                variable  value    model          ⋯
         │ Base.UUID                             String    Float64  Tuple…         ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ f4988dea-f42e-11ee-1052-e5336b447160  distance  4.38877  (:Linear, Flux ⋯
       2 │ f4b4f4d0-f42e-11ee-3bf1-b5ac117b9715  distance  4.17021  (:Linear, Flux
       3 │ f4b4f598-f42e-11ee-28f4-3d100497acf3  distance  4.31145  (:Linear, Flux
       4 │ f4b4f5c0-f42e-11ee-2888-b12fc3148ede  distance  4.17035  (:Linear, Flux
       5 │ f4b4f5de-f42e-11ee-1d74-15cc131d8fe4  distance  5.73182  (:MLP, FluxMod ⋯
       6 │ f4b4f5fc-f42e-11ee-0333-23e56121d20a  distance  5.50606  (:MLP, FluxMod
       7 │ f4b4f618-f42e-11ee-136e-658c23224dd0  distance  5.2114   (:MLP, FluxMod
       8 │ f4b4f638-f42e-11ee-2415-b704dd7602e6  distance  5.3623   (:MLP, FluxMod
                                                                   2 columns omitted

### Everything at once

Researchers, in particular, may be interested in combining all steps into one. This is the second scenario of Ad Hoc Benchmarking:

1.  Fit some machine learning model(s), generate counterfactual explanations and benchmark them.

It involves calling `benchmark` directly on counterfactual data (the only positional argument):

``` julia
bmk = benchmark(counterfactual_data)
```

This will use the default models from [`standard_models_catalogue`](@ref) and train them on the data. All available generators from [`generator_catalogue`](@ref) will also be used:

``` julia
@chain bmk() begin
    @filter(variable == "validity")
    @select(sample, variable, value, model, generator)
end
```

    200×5 DataFrame
     Row │ sample                                variable  value    model   genera ⋯
         │ Base.UUID                             String    Float64  Symbol  Symbol ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  gravit ⋯
       2 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  growin
       3 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  revise
       4 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  clue
       5 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  probe  ⋯
       6 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  dice
       7 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  clapro
       8 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  wachte
       9 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  generi ⋯
      10 │ fa9efec2-f42e-11ee-3ffa-8573e6574f34  validity      1.0  Linear  greedy
      11 │ fa9fcebc-f42e-11ee-12fc-fff8c5c8948e  validity      1.0  Linear  gravit
      ⋮  │                  ⋮                       ⋮         ⋮       ⋮            ⋱
     191 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     gravit
     192 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     growin ⋯
     193 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     revise
     194 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     clue
     195 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     probe
     196 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     dice   ⋯
     197 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     clapro
     198 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     wachte
     199 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     generi
     200 │ fb4e9e6a-f42e-11ee-101c-1197127b452f  validity      1.0  MLP     greedy ⋯
                                                       1 column and 179 rows omitted

Optionally, you can instead provide a dictionary of `models` and `generators` as before. Each value in the `models` dictionary should be one of two things:

1.  Either be an object `M` of type [`AbstractFittedModel`](@ref) that implements the [`Models.train`](@ref) method.
2.  Or a `DataType` that can be called on [`CounterfactualData`](@ref) to create an object `M` as in (a).

## Multiple Datasets

Benchmarks are run on single instances of type [`CounterfactualData`](@ref). This is our design choice for two reasons:

1.  We want to avoid the loops inside the `benchmark` method(s) from getting too nested and convoluted.
2.  While it is straightforward to infer metadata for models and generators, this is not the case for datasets.

Fortunately, it is very easy to run benchmarks for multiple datasets anyway, since `Benchmark` instances can be concatenated. To see how, let’s consider an example involving multiple datasets, models and generators:

``` julia
# Data:
datasets = Dict(
    :moons => CounterfactualData(load_moons()...),
    :circles => CounterfactualData(load_circles()...),
)

# Models:
models = Dict(
    :MLP => FluxModel,
    :Linear => Linear,
)

# Generators:
generators = Dict(
    :Generic => GenericGenerator(),
    :Greedy => GreedyGenerator(),
)
```

Then we can simply loop over the datasets and eventually concatenate the results like so:

``` julia
using CounterfactualExplanations.Evaluation: distance_measures
bmks = []
for (dataname, dataset) in datasets
    bmk = benchmark(dataset; models=models, generators=generators, measure=distance_measures)
    push!(bmks, bmk)
end
bmk = vcat(bmks[1], bmks[2]; ids=collect(keys(datasets)))
```

When `ids` are supplied, then a new id column is added to the evaluation data frame that contains unique identifiers for the different benchmarks. The optional `idcol_name` argument can be used to specify the name for that indicator column (defaults to `"dataset"`):

``` julia
@chain bmk() begin
    @group_by(dataset, generator)
    @filter(model == :MLP)
    @filter(variable == "distance_l1")
    @summarize(L1_norm=mean(value))
    @ungroup
end
```

    4×3 DataFrame
     Row │ dataset  generator  L1_norm  
         │ Symbol   Symbol     Float32  
    ─────┼──────────────────────────────
       1 │ moons    Generic    1.56555
       2 │ moons    Greedy     0.819269
       3 │ circles  Generic    1.83524
       4 │ circles  Greedy     0.498953
