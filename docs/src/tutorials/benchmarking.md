# Performance Benchmarks

``` @meta
CurrentModule = CounterfactualExplanations 
```

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
     Row │ sample  variable    value     generator                          model  ⋯
         │ Int64   String      Float64   Symbol                             Symbol ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │      1  distance    0.560887  GradientBasedGenerator(nothing, …  FluxMo ⋯
       2 │      1  redundancy  0.0       GradientBasedGenerator(nothing, …  FluxMo
       3 │      1  validity    1.0       GradientBasedGenerator(nothing, …  FluxMo
       4 │      2  distance    1.03898   GradientBasedGenerator(nothing, …  FluxMo
       5 │      2  redundancy  0.0       GradientBasedGenerator(nothing, …  FluxMo ⋯
       6 │      2  validity    0.8       GradientBasedGenerator(nothing, …  FluxMo
       7 │      3  distance    0.326338  GradientBasedGenerator(nothing, …  FluxMo
       8 │      3  redundancy  0.0       GradientBasedGenerator(nothing, …  FluxMo
       9 │      3  validity    1.0       GradientBasedGenerator(nothing, …  FluxMo ⋯
      10 │      4  distance    1.5446    GradientBasedGenerator(nothing, …  FluxMo
      11 │      4  redundancy  0.0       GradientBasedGenerator(nothing, …  FluxMo
      12 │      4  validity    1.0       GradientBasedGenerator(nothing, …  FluxMo
      13 │      5  distance    0.600052  GradientBasedGenerator(nothing, …  FluxMo ⋯
      14 │      5  redundancy  0.0       GradientBasedGenerator(nothing, …  FluxMo
      15 │      5  validity    0.8       GradientBasedGenerator(nothing, …  FluxMo
                                                                   3 columns omitted

To retrieve the granular dataset, simply do:

``` julia
bmk(agg=nothing)
```

    75×8 DataFrame
     Row │ sample  num_counterfactual  variable    value     generator             ⋯
         │ Int64   Int64               String      Float64   Symbol                ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │      1                   1  distance    0.562579  GradientBasedGenerato ⋯
       2 │      1                   2  distance    0.576755  GradientBasedGenerato
       3 │      1                   3  distance    0.530911  GradientBasedGenerato
       4 │      1                   4  distance    0.505757  GradientBasedGenerato
       5 │      1                   5  distance    0.628436  GradientBasedGenerato ⋯
       6 │      1                   1  redundancy  0.0       GradientBasedGenerato
       7 │      1                   2  redundancy  0.0       GradientBasedGenerato
       8 │      1                   3  redundancy  0.0       GradientBasedGenerato
       9 │      1                   4  redundancy  0.0       GradientBasedGenerato ⋯
      10 │      1                   5  redundancy  0.0       GradientBasedGenerato
      11 │      1                   1  validity    1.0       GradientBasedGenerato
      ⋮  │   ⋮             ⋮               ⋮          ⋮                      ⋮     ⋱
      66 │      5                   1  redundancy  0.0       GradientBasedGenerato
      67 │      5                   2  redundancy  0.0       GradientBasedGenerato ⋯
      68 │      5                   3  redundancy  0.0       GradientBasedGenerato
      69 │      5                   4  redundancy  0.0       GradientBasedGenerato
      70 │      5                   5  redundancy  0.0       GradientBasedGenerato
      71 │      5                   1  validity    1.0       GradientBasedGenerato ⋯
      72 │      5                   2  validity    1.0       GradientBasedGenerato
      73 │      5                   3  validity    0.0       GradientBasedGenerato
      74 │      5                   4  validity    1.0       GradientBasedGenerato
      75 │      5                   5  validity    1.0       GradientBasedGenerato ⋯
                                                       4 columns and 54 rows omitted

Since benchmarks return a `DataFrame` object on call, post-processing is straightforward. For example, we could use [`Tidier.jl`](https://kdpsingh.github.io/Tidier.jl/dev/):

``` julia
using Tidier
@chain bmk() begin
    @filter(variable == "distance")
    @select(sample, variable, value)
end
```

    5×3 DataFrame
     Row │ sample  variable  value    
         │ Int64   String    Float64  
    ─────┼────────────────────────────
       1 │      1  distance  0.560887
       2 │      2  distance  1.03898
       3 │      3  distance  0.326338
       4 │      4  distance  1.5446
       5 │      5  distance  0.600052

### Metadata for Counterfactual Explanations

Benchmarks always report metadata for each counterfactual explanation, which is automatically inferred by default. The default metadata concerns the explained `model` and the employed `generator`. In the current example, we used the same model and generator for each individual:

``` julia
@chain bmk() begin
    @group_by(sample)
    @select(sample, model, generator)
    @summarize(model=unique(model),generator=unique(generator))
    @ungroup
end
```

    5×3 DataFrame
     Row │ sample  model                              generator                    ⋯
         │ Int64   Symbol                             Symbol                       ⋯
    ─────┼──────────────────────────────────────────────────────────────────────────
       1 │      1  FluxModel(Chain(Dense(2 => 2)), …  GradientBasedGenerator(nothi ⋯
       2 │      2  FluxModel(Chain(Dense(2 => 2)), …  GradientBasedGenerator(nothi
       3 │      3  FluxModel(Chain(Dense(2 => 2)), …  GradientBasedGenerator(nothi
       4 │      4  FluxModel(Chain(Dense(2 => 2)), …  GradientBasedGenerator(nothi
       5 │      5  FluxModel(Chain(Dense(2 => 2)), …  GradientBasedGenerator(nothi ⋯
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
    @summarize(model=unique(model),generator=unique(generator))
    @ungroup
end
```

    5×3 DataFrame
     Row │ sample  model   generator 
         │ Int64   String  String    
    ─────┼───────────────────────────
       1 │      1  MLP     Generic
       2 │      2  MLP     Generic
       3 │      3  MLP     Generic
       4 │      4  MLP     Generic
       5 │      5  MLP     Generic

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
     Row │ sample  variable  value     model   generator     
         │ Int64   String    Float64   Symbol  Symbol        
    ─────┼───────────────────────────────────────────────────
       1 │      1  distance  0.346822  Linear  Gravitational
       2 │      1  distance  0.45062   Linear  ClaPROAR
       3 │      1  distance  0.42487   Linear  Generic
       4 │      1  distance  0.486442  Linear  Wachter
       5 │      1  distance  0.687121  MLP     Gravitational
       6 │      1  distance  0.728272  MLP     ClaPROAR
       7 │      1  distance  0.57091   MLP     Generic
       8 │      1  distance  0.711715  MLP     Wachter

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

    165×5 DataFrame
     Row │ sample  variable  value    model   generator       
         │ Int64   String    Float64  Symbol  Symbol          
    ─────┼────────────────────────────────────────────────────
       1 │      1  validity      1.0  Linear  gravitational
       2 │      2  validity      1.0  Linear  gravitational
       3 │      3  validity      1.0  Linear  gravitational
       4 │      4  validity      1.0  Linear  gravitational
       5 │      5  validity      1.0  Linear  gravitational
       6 │      1  validity      1.0  Linear  growing_spheres
       7 │      2  validity      1.0  Linear  growing_spheres
       8 │      3  validity      1.0  Linear  growing_spheres
       9 │      4  validity      1.0  Linear  growing_spheres
      10 │      5  validity      1.0  Linear  growing_spheres
      11 │      1  validity      1.0  Linear  revise
      ⋮  │   ⋮        ⋮         ⋮       ⋮            ⋮
     156 │     11  validity      1.0  MLP     generic
     157 │     12  validity      1.0  MLP     generic
     158 │     13  validity      1.0  MLP     generic
     159 │     14  validity      0.0  MLP     generic
     160 │     15  validity      1.0  MLP     generic
     161 │     11  validity      0.0  MLP     greedy
     162 │     12  validity      1.0  MLP     greedy
     163 │     13  validity      1.0  MLP     greedy
     164 │     14  validity      0.0  MLP     greedy
     165 │     15  validity      1.0  MLP     greedy
                                              144 rows omitted

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
    :moons => load_moons(),
    :circles => load_circles(),
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
    bmk = benchmark(dataset; models=models, generators=generators, measure=distance_measures, verbose=true)
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
       1 │ circles  Generic    1.88603
       2 │ circles  Greedy     0.66296
       3 │ moons    Generic    0.787587
       4 │ moons    Greedy     0.612045
