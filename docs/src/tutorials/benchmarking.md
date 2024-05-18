

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

Under the hood, the [`benchmark(counterfactual_explanations::Vector{CounterfactualExplanation})`](@ref) uses [`CounterfactualExplanations.Evaluation.evaluate(ce::CounterfactualExplanation)`](@ref) to generate a [`Benchmark`](@ref) object, which contains the evaluation in its most granular form as a `DataFrame`.

### Working with `Benchmark`s

For convenience, the `DataFrame` containing the evaluation can be returned by simply calling the `Benchmark` object. By default, the aggregated evaluation measures across `id` (in line with the default behaviour of `evaluate`).

``` julia
bmk()
```

    15×7 DataFrame
     Row │ sample                                variable    value    generator    ⋯
         │ Base.UUID                             String      Float64  Symbol       ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ 239104d0-f59f-11ee-3d0c-d1db071927ff  distance    3.17243  GradientBase ⋯
       2 │ 239104d0-f59f-11ee-3d0c-d1db071927ff  redundancy  0.0      GradientBase
       3 │ 239104d0-f59f-11ee-3d0c-d1db071927ff  validity    1.0      GradientBase
       4 │ 2398b3e2-f59f-11ee-3323-13d53fb7e75b  distance    3.07148  GradientBase
       5 │ 2398b3e2-f59f-11ee-3323-13d53fb7e75b  redundancy  0.0      GradientBase ⋯
       6 │ 2398b3e2-f59f-11ee-3323-13d53fb7e75b  validity    1.0      GradientBase
       7 │ 2398b916-f59f-11ee-3f13-bd00858a39af  distance    3.62159  GradientBase
       8 │ 2398b916-f59f-11ee-3f13-bd00858a39af  redundancy  0.0      GradientBase
       9 │ 2398b916-f59f-11ee-3f13-bd00858a39af  validity    1.0      GradientBase ⋯
      10 │ 2398bce8-f59f-11ee-37c1-ef7c6de27b6b  distance    2.62783  GradientBase
      11 │ 2398bce8-f59f-11ee-37c1-ef7c6de27b6b  redundancy  0.0      GradientBase
      12 │ 2398bce8-f59f-11ee-37c1-ef7c6de27b6b  validity    1.0      GradientBase
      13 │ 2398c08a-f59f-11ee-175b-81c155750752  distance    2.91985  GradientBase ⋯
      14 │ 2398c08a-f59f-11ee-175b-81c155750752  redundancy  0.0      GradientBase
      15 │ 2398c08a-f59f-11ee-175b-81c155750752  validity    1.0      GradientBase
                                                                   4 columns omitted

To retrieve the granular dataset, simply do:

``` julia
bmk(agg=nothing)
```

    75×8 DataFrame
     Row │ sample                                num_counterfactual  variable    v ⋯
         │ Base.UUID                             Int64               String      F ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   1  distance    3 ⋯
       2 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   2  distance    3
       3 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   3  distance    3
       4 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   4  distance    3
       5 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   5  distance    3 ⋯
       6 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   1  redundancy  0
       7 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   2  redundancy  0
       8 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   3  redundancy  0
       9 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   4  redundancy  0 ⋯
      10 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   5  redundancy  0
      11 │ 239104d0-f59f-11ee-3d0c-d1db071927ff                   1  validity    1
      ⋮  │                  ⋮                            ⋮               ⋮         ⋱
      66 │ 2398c08a-f59f-11ee-175b-81c155750752                   1  redundancy  0
      67 │ 2398c08a-f59f-11ee-175b-81c155750752                   2  redundancy  0 ⋯
      68 │ 2398c08a-f59f-11ee-175b-81c155750752                   3  redundancy  0
      69 │ 2398c08a-f59f-11ee-175b-81c155750752                   4  redundancy  0
      70 │ 2398c08a-f59f-11ee-175b-81c155750752                   5  redundancy  0
      71 │ 2398c08a-f59f-11ee-175b-81c155750752                   1  validity    1 ⋯
      72 │ 2398c08a-f59f-11ee-175b-81c155750752                   2  validity    1
      73 │ 2398c08a-f59f-11ee-175b-81c155750752                   3  validity    1
      74 │ 2398c08a-f59f-11ee-175b-81c155750752                   4  validity    1
      75 │ 2398c08a-f59f-11ee-175b-81c155750752                   5  validity    1 ⋯
                                                       5 columns and 54 rows omitted

Since benchmarks return a `DataFrame` object on call, post-processing is straightforward. For example, we could use [`Tidier.jl`](https://kdpsingh.github.io/Tidier.jl/dev/):

``` julia
using Tidier
@chain bmk() begin
    @filter(variable == "distance")
    @select(sample, variable, value)
end
```

    5×3 DataFrame
     Row │ sample                                variable  value   
         │ Base.UUID                             String    Float64 
    ─────┼─────────────────────────────────────────────────────────
       1 │ 239104d0-f59f-11ee-3d0c-d1db071927ff  distance  3.17243
       2 │ 2398b3e2-f59f-11ee-3323-13d53fb7e75b  distance  3.07148
       3 │ 2398b916-f59f-11ee-3f13-bd00858a39af  distance  3.62159
       4 │ 2398bce8-f59f-11ee-37c1-ef7c6de27b6b  distance  2.62783
       5 │ 2398c08a-f59f-11ee-175b-81c155750752  distance  2.91985

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
       1 │ 239104d0-f59f-11ee-3d0c-d1db071927ff  FluxModel(Chain(Dense(2 => 2)), … ⋯
       2 │ 2398b3e2-f59f-11ee-3323-13d53fb7e75b  FluxModel(Chain(Dense(2 => 2)), …
       3 │ 2398b916-f59f-11ee-3f13-bd00858a39af  FluxModel(Chain(Dense(2 => 2)), …
       4 │ 2398bce8-f59f-11ee-37c1-ef7c6de27b6b  FluxModel(Chain(Dense(2 => 2)), …
       5 │ 2398c08a-f59f-11ee-175b-81c155750752  FluxModel(Chain(Dense(2 => 2)), … ⋯
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
       1 │ 27fae496-f59f-11ee-2c30-f35d1025a6d4  MLP     Generic
       2 │ 27fdcc6a-f59f-11ee-030b-152c9794c5f1  MLP     Generic
       3 │ 27fdd04a-f59f-11ee-2010-e1732ff5d8d2  MLP     Generic
       4 │ 27fdd340-f59f-11ee-1d20-050a69dcacef  MLP     Generic
       5 │ 27fdd5fc-f59f-11ee-02e8-d198e436abb3  MLP     Generic

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
       1 │ 2cba5eee-f59f-11ee-1844-cbc7a8372a38  distance  4.38877  (:Linear, Flux ⋯
       2 │ 2cd740fe-f59f-11ee-35c3-1157eb1b7583  distance  4.17021  (:Linear, Flux
       3 │ 2cd741e2-f59f-11ee-2b09-0d55ef9892b9  distance  4.31145  (:Linear, Flux
       4 │ 2cd7420c-f59f-11ee-1996-6fa75e23bb57  distance  4.17035  (:Linear, Flux
       5 │ 2cd74234-f59f-11ee-0ad0-9f21949f5932  distance  5.73182  (:MLP, FluxMod ⋯
       6 │ 2cd7425c-f59f-11ee-3eb4-af34f85ffd3d  distance  5.50606  (:MLP, FluxMod
       7 │ 2cd7427a-f59f-11ee-10d3-a1df6c8dc125  distance  5.2114   (:MLP, FluxMod
       8 │ 2cd74298-f59f-11ee-32d1-f501c104fea8  distance  5.3623   (:MLP, FluxMod
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
       1 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  gravit ⋯
       2 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  growin
       3 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  revise
       4 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  clue
       5 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  probe  ⋯
       6 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  dice
       7 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  clapro
       8 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  wachte
       9 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  generi ⋯
      10 │ 32d1817e-f59f-11ee-152f-a30b18c2e6f7  validity      1.0  Linear  greedy
      11 │ 32d255e8-f59f-11ee-3e8d-a9e9f6e23ea8  validity      1.0  Linear  gravit
      ⋮  │                  ⋮                       ⋮         ⋮       ⋮            ⋱
     191 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     gravit
     192 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     growin ⋯
     193 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     revise
     194 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     clue
     195 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     probe
     196 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     dice   ⋯
     197 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     clapro
     198 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     wachte
     199 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     generi
     200 │ 3382d08a-f59f-11ee-10b3-f7d18cf7d3b5  validity      1.0  MLP     greedy ⋯
                                                       1 column and 179 rows omitted

Optionally, you can instead provide a dictionary of `models` and `generators` as before. Each value in the `models` dictionary should be one of two things:

1.  Either be an object `M` of type [`AbstractModel`](@ref) that implements the [`Models.train`](@ref) method.
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
