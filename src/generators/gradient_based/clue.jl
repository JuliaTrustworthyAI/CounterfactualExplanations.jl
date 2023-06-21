"""
    CLUEGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        ϵ::AbstractFloat=0.1,
        τ::AbstractFloat=1e-5
    )
An outer constructor method that instantiates a CLUE generator.
# Examples
```julia-repl
generator = CLUEGenerator()
```
"""
function CLUEGenerator(; λ::AbstractFloat=0.1, latent_space=true, kwargs...)
    return GradientBasedGenerator(;
        loss=predictive_entropy,
        penalty=default_distance,
        λ=λ,
        latent_space=latent_space,
        kwargs...,
    )
end

function predictive_entropy(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)
    model = ce.M
    counterfactual_data = ce.data
    X = CounterfactualExplanations.decode_state(ce)
    p = CounterfactualExplanations.Models.predict_proba(model, counterfactual_data, X)
    output = agg(sum(@.(p * log(p)); dims=2))
    return output
end
