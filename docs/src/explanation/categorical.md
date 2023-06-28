# Categorical Features

``` @meta
CurrentModule = CounterfactualExplanations 
```

To illustrate how data is preprocessed under the hood, we consider a simple toy dataset with three categorical features (`name`, `grade` and `sex`) and one continuous feature (`age`):

``` julia
X = (
    name=categorical(["Danesh", "Lee", "Mary", "John"]),
    grade=categorical(["A", "B", "A", "C"], ordered=true),
    sex=categorical(["male","female","male","male"]),
    height=[1.85, 1.67, 1.5, 1.67],
)
schema(X)
```

Categorical features are expected to be one-hot or dummy encoded. To this end, we could use `MLJ`, for example:

``` julia
hot = OneHotEncoder()
mach = fit!(machine(hot, X))
W = transform(mach, X)
schema(W)
```

    ┌──────────────┬────────────┬─────────┐
    │ names        │ scitypes   │ types   │
    ├──────────────┼────────────┼─────────┤
    │ name__Danesh │ Continuous │ Float64 │
    │ name__John   │ Continuous │ Float64 │
    │ name__Lee    │ Continuous │ Float64 │
    │ name__Mary   │ Continuous │ Float64 │
    │ grade__A     │ Continuous │ Float64 │
    │ grade__B     │ Continuous │ Float64 │
    │ grade__C     │ Continuous │ Float64 │
    │ sex__female  │ Continuous │ Float64 │
    │ sex__male    │ Continuous │ Float64 │
    │ height       │ Continuous │ Float64 │
    └──────────────┴────────────┴─────────┘

The matrix that will be perturbed during the counterfactual search looks as follows:

``` julia
X = permutedims(MLJBase.matrix(W))
```

    10×4 Matrix{Float64}:
     1.0   0.0   0.0  0.0
     0.0   0.0   0.0  1.0
     0.0   1.0   0.0  0.0
     0.0   0.0   1.0  0.0
     1.0   0.0   1.0  0.0
     0.0   1.0   0.0  0.0
     0.0   0.0   0.0  1.0
     0.0   1.0   0.0  0.0
     1.0   0.0   1.0  1.0
     1.85  1.67  1.5  1.67

The `CounterfactualData` constructor takes two optional arguments that can be used to specify the indices of categorical and continuous features. If nothing is supplied, all features are assumed to be continuous. For categorical features, the constructor expects and array of arrays of integers (`Vector{Vector{Int}}`) where each subarray includes the indices of a all one-hot encoded rows related to a single categorical feature. In the example above, the `name` feature is one-hot encoded across rows 1, 2 and 3 of `X`.

``` julia
features_categorical = [
    [1,2,3,4],    # name
    [5,6,7],    # grade
    [8,9]       # sex
]
features_continuous = [10]
```

We propose the following simple logic for reconstructing categorical encodings after perturbations:

- For one-hot encoded features with multiple classes, choose the maximum.
- For binary features, clip the perturbed value to fall into $[0,1]$ and round to the nearest of the two integers.

``` julia
function reconstruct_cat_encoding(x)
    map(features_categorical) do cat_group_index
        if length(cat_group_index) > 1
            x[cat_group_index] = Int.(x[cat_group_index] .== maximum(x[cat_group_index]))
            if sum(x[cat_group_index]) > 1
                ties = findall(x[cat_group_index] .== 1)
                _x = zeros(length(x[cat_group_index]))
                winner = rand(ties,1)[1]
                _x[winner] = 1
                x[cat_group_index] = _x
            end
        else
            x[cat_group_index] = [round(clamp(x[cat_group_index][1],0,1))]
        end
    end
    return x
end
```

Let’s look at a few simple examples to see how this function works. Firstly, consider the case of perturbing a single element:

``` julia
x = X[:,1]
x[1] = 1.1
x
```

    10-element Vector{Float64}:
     1.1
     0.0
     0.0
     0.0
     1.0
     0.0
     0.0
     0.0
     1.0
     1.85

The reconstructed one-hot-encoded vector will look like this:

``` julia
reconstruct_cat_encoding(x)
```

    10-element Vector{Float64}:
     1.0
     0.0
     0.0
     0.0
     1.0
     0.0
     0.0
     0.0
     1.0
     1.85

Next, consider the case of perturbing multiple elements:

``` julia
x[2] = 1.1
x[3] = -1.2
x
```

    10-element Vector{Float64}:
      1.0
      1.1
     -1.2
      0.0
      1.0
      0.0
      0.0
      0.0
      1.0
      1.85

The reconstructed one-hot-encoded vector will look like this:

``` julia
reconstruct_cat_encoding(x)
```

    10-element Vector{Float64}:
     0.0
     1.0
     0.0
     0.0
     1.0
     0.0
     0.0
     0.0
     1.0
     1.85

Finally, let’s introduce a tie:

``` julia
x[1] = 1.0
x
```

    10-element Vector{Float64}:
     1.0
     1.0
     0.0
     0.0
     1.0
     0.0
     0.0
     0.0
     1.0
     1.85

The reconstructed one-hot-encoded vector will look like this:

``` julia
reconstruct_cat_encoding(x)
```

    10-element Vector{Float64}:
     1.0
     0.0
     0.0
     0.0
     1.0
     0.0
     0.0
     0.0
     1.0
     1.85
