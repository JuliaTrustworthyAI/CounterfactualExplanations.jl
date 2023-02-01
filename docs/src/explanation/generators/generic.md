
``` @meta
CurrentModule = CounterfactualExplanations 
```

# `GenericGenerator`

We use the term **generic** to relate to the counterfactual generator proposed by Wachter, Mittelstadt, and Russell (2017).

## Description

As the term indicates, this approach is simple: it forms the baseline approach for gradient-based counterfactual generators. Wachter, Mittelstadt, and Russell (2017) were among the first to realise that

> \[…\] explanations can, in principle, be offered without opening the “black box.”
>
> — Wachter, Mittelstadt, and Russell (2017)

Gradient descent is performed directly in the feature space. Concerning the cost heuristic, the authors choose to penalize the distance of counterfactuals from the factual value. This is based on the intuitive notion that larger feature perturbations require greater effort.

## Usage

The approach can be used in our package as follows:

``` julia
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
plot(ce)
```

![](generic_files/figure-commonmark/cell-3-output-1.svg)

## References

Wachter, Sandra, Brent Mittelstadt, and Chris Russell. 2017. “Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR.” *Harv. JL & Tech.* 31: 841.
