"""
	generate_counterfactual(
        x::Matrix,
        target::RawTargetType,
        data::CounterfactualData,
        M::Models.AbstractModel,
        generator::AbstractGenerator;
        num_counterfactuals::Int=1,
        initialization::Symbol=:add_perturbation,
        convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
        timeout::Union{Nothing,Real}=nothing,
    )

The core function that is used to run counterfactual search for a given factual `x`, target, counterfactual data, model and generator. Keywords can be used to specify the desired threshold for the predicted target class probability and the maximum number of iterations.

# Arguments

- `x::Matrix`: Factual data point.
- `target::RawTargetType`: Target class.
- `data::CounterfactualData`: Counterfactual data.
- `M::Models.AbstractModel`: Fitted model.
- `generator::AbstractGenerator`: Generator.
- `num_counterfactuals::Int=1`: Number of counterfactuals to generate for factual.
- `initialization::Symbol=:add_perturbation`: Initialization method. By default, the initialization is done by adding a small random perturbation to the factual to achieve more robustness.
- `convergence::Union{AbstractConvergence,Symbol}=:decision_threshold`: Convergence criterion. By default, the convergence is based on the decision threshold. Possible values are `:decision_threshold`, `:max_iter`, `:generator_conditions` or a conrete convergence object (e.g. [`DecisionThresholdConvergence`](@ref)). 
- `timeout::Union{Nothing,Int}=nothing`: Timeout in seconds.

# Examples

## Generic generator

```jldoctest
julia> using CounterfactualExplanations

julia> using TaijaData
       
        # Counteractual data and model:

julia> counterfactual_data = CounterfactualData(load_linearly_separable()...);

julia> M = fit_model(counterfactual_data, :Linear);

julia> target = 2;

julia> factual = 1;

julia> chosen = rand(findall(predict_label(M, counterfactual_data) .== factual));

julia> x = select_factual(counterfactual_data, chosen);
       
       # Search:

julia> generator = Generators.GenericGenerator();

julia> ce = generate_counterfactual(x, target, counterfactual_data, M, generator);

julia> converged(ce.convergence, ce)
true
```

## Broadcasting

The `generate_counterfactual` method can also be broadcasted over a tuple containing an array. This allows for generating multiple counterfactuals in parallel. 

```jldoctest
julia> chosen = rand(findall(predict_label(M, counterfactual_data) .== factual), 5);

julia> xs = select_factual(counterfactual_data, chosen);

julia> ces = generate_counterfactual.(xs, target, counterfactual_data, M, generator);

julia> converged(ce.convergence, ce)
true
```
"""
function generate_counterfactual(
    x::Matrix,
    target::RawTargetType,
    data::CounterfactualData,
    M::Models.AbstractModel,
    generator::AbstractGenerator;
    num_counterfactuals::Int=1,
    initialization::Symbol=:add_perturbation,
    convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
    timeout::Union{Nothing,Real}=nothing,
    return_flattened::Bool=false
)

    output(ce::CounterfactualExplanation) = return_flattened ? flatten(ce) : ce

    # Initialize:
    ce = CounterfactualExplanation(
        x,
        target,
        data,
        M,
        generator;
        num_counterfactuals=num_counterfactuals,
        initialization=initialization,
        convergence=convergence,
    )

    # Check for redundancy (assess if already converged with respect to factual):
    if Convergence.converged(ce.convergence, ce, ce.factual)
        @info "Factual already in target class and probability exceeds threshold."
        return output(ce)
    end

    # Check for incompatibility:
    if Generators.incompatible(ce.generator, ce)
        @info "Generator is incompatible with other specifications for the counterfactual explanation (e.g. the model). See warnings for details. No search completed."
        return output(ce)
    end

    # Search:
    timer = isnothing(timeout) ? nothing : Timer(timeout)
    while !terminated(ce)
        CounterfactualExplanations.update!(ce)
        if !isnothing(timer)
            yield()
            if !isopen(timer)
                @info "Counterfactual search timed out before convergence"
                break
            end
        end
    end

    # Return full or flattened explanation:
    return output(ce)
end

"""
    generate_counterfactual(x::Tuple{<:AbstractArray}, args...; kwargs...)

Overloads the `generate_counterfactual` method to accept a tuple containing and array. This allows for broadcasting over `Zip` iterators.
"""
function generate_counterfactual(x::Tuple{<:AbstractArray}, args...; kwargs...)
    return generate_counterfactual(x[1], args...; kwargs...)
end
