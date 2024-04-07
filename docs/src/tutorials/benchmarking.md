

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
       1 │ 4d870646-f4c2-11ee-2da2-038400ab9389  distance    3.17243  GradientBase ⋯
       2 │ 4d870646-f4c2-11ee-2da2-038400ab9389  redundancy  0.0      GradientBase
       3 │ 4d870646-f4c2-11ee-2da2-038400ab9389  validity    1.0      GradientBase
       4 │ 4d8e474e-f4c2-11ee-20a0-271d06823dfc  distance    3.07148  GradientBase
       5 │ 4d8e474e-f4c2-11ee-20a0-271d06823dfc  redundancy  0.0      GradientBase ⋯
       6 │ 4d8e474e-f4c2-11ee-20a0-271d06823dfc  validity    1.0      GradientBase
       7 │ 4d8e4c80-f4c2-11ee-1287-81610e5fb51c  distance    3.62159  GradientBase
       8 │ 4d8e4c80-f4c2-11ee-1287-81610e5fb51c  redundancy  0.0      GradientBase
       9 │ 4d8e4c80-f4c2-11ee-1287-81610e5fb51c  validity    1.0      GradientBase ⋯
      10 │ 4d8e5068-f4c2-11ee-11cc-193c402ab999  distance    2.62783  GradientBase
      11 │ 4d8e5068-f4c2-11ee-11cc-193c402ab999  redundancy  0.0      GradientBase
      12 │ 4d8e5068-f4c2-11ee-11cc-193c402ab999  validity    1.0      GradientBase
      13 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f  distance    2.91985  GradientBase ⋯
      14 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f  redundancy  0.0      GradientBase
      15 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f  validity    1.0      GradientBase
                                                                   4 columns omitted

To retrieve the granular dataset, simply do:

``` julia
bmk(agg=nothing)
```

    75×8 DataFrame
     Row │ sample                                num_counterfactual  variable    v ⋯
         │ Base.UUID                             Int64               String      F ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   1  distance    3 ⋯
       2 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   2  distance    3
       3 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   3  distance    3
       4 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   4  distance    3
       5 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   5  distance    3 ⋯
       6 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   1  redundancy  0
       7 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   2  redundancy  0
       8 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   3  redundancy  0
       9 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   4  redundancy  0 ⋯
      10 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   5  redundancy  0
      11 │ 4d870646-f4c2-11ee-2da2-038400ab9389                   1  validity    1
      ⋮  │                  ⋮                            ⋮               ⋮         ⋱
      66 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   1  redundancy  0
      67 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   2  redundancy  0 ⋯
      68 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   3  redundancy  0
      69 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   4  redundancy  0
      70 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   5  redundancy  0
      71 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   1  validity    1 ⋯
      72 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   2  validity    1
      73 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   3  validity    1
      74 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   4  validity    1
      75 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f                   5  validity    1 ⋯
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
       1 │ 4d870646-f4c2-11ee-2da2-038400ab9389  distance  3.17243
       2 │ 4d8e474e-f4c2-11ee-20a0-271d06823dfc  distance  3.07148
       3 │ 4d8e4c80-f4c2-11ee-1287-81610e5fb51c  distance  3.62159
       4 │ 4d8e5068-f4c2-11ee-11cc-193c402ab999  distance  2.62783
       5 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f  distance  2.91985

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
       1 │ 4d870646-f4c2-11ee-2da2-038400ab9389  FluxModel(Chain(Dense(2 => 2)), … ⋯
       2 │ 4d8e474e-f4c2-11ee-20a0-271d06823dfc  FluxModel(Chain(Dense(2 => 2)), …
       3 │ 4d8e4c80-f4c2-11ee-1287-81610e5fb51c  FluxModel(Chain(Dense(2 => 2)), …
       4 │ 4d8e5068-f4c2-11ee-11cc-193c402ab999  FluxModel(Chain(Dense(2 => 2)), …
       5 │ 4d8e53f6-f4c2-11ee-2cd8-d5ee288aa79f  FluxModel(Chain(Dense(2 => 2)), … ⋯
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
       1 │ 51c54c68-f4c2-11ee-08c0-d9c30ebb747b  MLP     Generic
       2 │ 51c838b0-f4c2-11ee-2d0c-0733278674ca  MLP     Generic
       3 │ 51c83c84-f4c2-11ee-17ed-d9ba1ae239c3  MLP     Generic
       4 │ 51c83fea-f4c2-11ee-32bf-a552d6fa02bb  MLP     Generic
       5 │ 51c842e2-f4c2-11ee-1ebf-ed383d576cc8  MLP     Generic

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
       1 │ 56537c94-f4c2-11ee-28a2-33f830cc2aeb  distance  4.38877  (:Linear, Flux ⋯
       2 │ 566e1272-f4c2-11ee-3976-55aee61c911f  distance  4.17021  (:Linear, Flux
       3 │ 566e1330-f4c2-11ee-22a2-916be39f526d  distance  4.31145  (:Linear, Flux
       4 │ 566e135a-f4c2-11ee-0486-051aa65063ec  distance  4.17035  (:Linear, Flux
       5 │ 566e1380-f4c2-11ee-2594-0f37b28bb409  distance  5.73182  (:MLP, FluxMod ⋯
       6 │ 566e139e-f4c2-11ee-32a1-df1a7abb4111  distance  5.50606  (:MLP, FluxMod
       7 │ 566e13ba-f4c2-11ee-1c05-81ec3ba1ff08  distance  5.2114   (:MLP, FluxMod
       8 │ 566e13da-f4c2-11ee-07b0-61ef5fc9258a  distance  5.3623   (:MLP, FluxMod
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
       1 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  gravit ⋯
       2 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  growin
       3 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  revise
       4 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  clue
       5 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  probe  ⋯
       6 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  dice
       7 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  clapro
       8 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  wachte
       9 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  generi ⋯
      10 │ 5c195b3c-f4c2-11ee-322a-9173776f3b7a  validity      1.0  Linear  greedy
      11 │ 5c1a277e-f4c2-11ee-0a8e-fb5f14a8e654  validity      1.0  Linear  gravit
      ⋮  │                  ⋮                       ⋮         ⋮       ⋮            ⋱
     191 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     gravit
     192 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     growin ⋯
     193 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     revise
     194 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     clue
     195 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     probe
     196 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     dice   ⋯
     197 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     clapro
     198 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     wachte
     199 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     generi
     200 │ 5cc2ba42-f4c2-11ee-3d2a-d5b8e088cf5b  validity      1.0  MLP     greedy ⋯
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
