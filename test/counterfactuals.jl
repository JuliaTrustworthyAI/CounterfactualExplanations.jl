# NOTE:
# This is probably the most important/useful test script, because it runs through the whole process of: 
# - loading artifacts
# - setting up counterfactual search for various models and generators
# - running counterfactual search

using CounterfactualExplanations
using CounterfactualExplanations.Models
using CounterfactualExplanations.Counterfactuals
using CounterfactualExplanations.Generators
using Random, LinearAlgebra, MLUtils, Flux
Random.seed!(1234)
max_reconstruction_error = 2.0

### Load synthetic data and models
synthetic = CounterfactualExplanations.Data.load_synthetic([:flux])

# Set up:
generators = Dict(
    :generic => Generators.GenericGenerator,
    :greedy => Generators.GreedyGenerator,
    :revise => Generators.REVISEGenerator,
    # :dice => Generators.DiCEGenerator
)

# # Quick one - Generic:
# generator = generators[:dice]()
# value = synthetic[:classification_binary]
# X, ys = (value[:data][:xs],value[:data][:ys])
# X = MLUtils.stack(X,dims=2)
# ys_cold = length(ys[1]) > 1 ? [Flux.onecold(y_,1:length(ys[1])) for y_ in ys] : ys
# counterfactual_data = CounterfactualData(X,ys')
# M = value[:models][:flux][:model]
# # Randomly selected factual:
# Random.seed!(123)
# x = select_factual(counterfactual_data,rand(1:size(X)[2]))
# γ = 0.9
# p_ = probs(M, x)
# if size(p_)[1] > 1
#     y = Flux.onecold(p_,unique(ys_cold))
#     target = rand(unique(ys_cold)[1:end .!= y]) # opposite label as target
# else
#     y = round(p_[1])
#     target = y ==0 ? 1 : 0 
# end
# counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator; γ=γ, num_counterfactuals=2)

# LOOP:
for (key, generator_) ∈ generators
    name = uppercasefirst(string(key))
    @testset "$name" begin
    
        # Generator:
        generator = generator_()

        @testset "Models for synthetic data" begin
        
            for (key, value) ∈ synthetic
                name = string(key)
                @testset "$name" begin
                    X, ys = (value[:data][:xs],value[:data][:ys])
                    X = MLUtils.stack(X,dims=2)
                    ys_cold = length(ys[1]) > 1 ? [Flux.onecold(y_,1:length(ys[1])) for y_ in ys] : ys
                    counterfactual_data = CounterfactualData(X,ys')
                    for (likelihood, model) ∈ value[:models]
                        name = string(likelihood)
                        @testset "$name" begin
                            M = model[:model]
                            # Randomly selected factual:
                            Random.seed!(123)
                            x = select_factual(counterfactual_data,rand(1:size(X)[2]))
                            
                            @testset "Predetermined outputs" begin
                                γ = 0.9
                                p_ = probs(M, x)
                                if size(p_)[1] > 1
                                    y = Flux.onecold(p_,unique(ys_cold))
                                    target = rand(unique(ys_cold)[1:end .!= y]) # opposite label as target
                                else
                                    y = round(p_[1])
                                    target = y ==0 ? 1 : 0 
                                end
                                counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator; γ=γ)
                                if typeof(generator) <: Generators.AbstractLatentSpaceGenerator
                                    @test counterfactual.latent_space
                                end
                                @test counterfactual.target == target
                                @test counterfactual.x == x && Counterfactuals.factual(counterfactual) == x
                                @test Counterfactuals.factual_label(counterfactual) == y
                                @test Counterfactuals.factual_probability(counterfactual) == p_
                            end
                    
                            @testset "Convergence" begin
                    
                                # Already in target and exceeding threshold probability:
                                p_ = probs(M, x)
                                if size(p_)[1] > 1
                                    y = Flux.onecold(p_,unique(ys_cold))[1]
                                    target = y
                                else
                                    target = round(p_[1])==0 ? 0 : 1 
                                end
                                counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator)
                                @test length(path(counterfactual))==1
                                if typeof(generator) <: Generators.AbstractLatentSpaceGenerator
                                    # In case of latent space search, there is a reconstruction error:
                                    @test maximum(abs.(counterfactual.x .- counterfactual.f(counterfactual.s′))) < max_reconstruction_error
                                else
                                    @test all(counterfactual.x .== counterfactual.f(counterfactual.s′))
                                end
                                @test converged(counterfactual)
                                @test Counterfactuals.terminated(counterfactual)
                                @test Counterfactuals.total_steps(counterfactual) == 1
                    
                                # Threshold reached if converged:
                                γ = 0.9
                                p_ = probs(M, x)
                                if size(p_)[1] > 1
                                    y = Flux.onecold(p_,unique(ys_cold))
                                    target = rand(unique(ys_cold)[1:end .!= y]) # opposite label as target
                                else
                                    target = round(p_[1])==0 ? 1 : 0 
                                end
                                T = 1000
                                counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator; γ=γ, T=T)
                                import CounterfactualExplanations.Counterfactuals: counterfactual_probability
                                @test !converged(counterfactual) || target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                                @test !converged(counterfactual) || length(path(counterfactual)) <= T
                    
                            end
                        end
                    end
                end
            end
            
        end
    
        @testset "LogisticModel" begin

            # Data:
            using CounterfactualExplanations.Data
            using Random
            Random.seed!(1234)
            N = 25
            xs, ys = Data.toy_data_linear(N)
            X = hcat(xs...)
            counterfactual_data = CounterfactualData(X,ys')

            # Model
            using CounterfactualExplanations.Models: LogisticModel, probs 
            # Logit model:
            w = [1.0 1.0] # true coefficients
            b = 0
            M = LogisticModel(w, [b])
    
            # Randomly selected factual:
            Random.seed!(123)
            x = select_factual(counterfactual_data,rand(1:size(X)[2]))
            y = round(probs(M, x)[1])
            
            @testset "Predetermined outputs" begin
                γ = 0.9
                target = round(probs(M, x)[1])==0 ? 1 : 0 
                counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator; γ=γ)
                @test counterfactual.target == target
                @test counterfactual.x == x
            end
    
            @testset "Convergence" begin
    
                # Already in target and exceeding threshold probability:
                γ = probs(M, x)[1]
                target = round(γ)
                counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator)
                @test length(path(counterfactual))==1
                if typeof(generator) <: Generators.AbstractLatentSpaceGenerator
                    # In case of latent space search, there is a reconstruction error:
                    @test maximum(abs.(counterfactual.x .- counterfactual.f(counterfactual.s′))) < max_reconstruction_error
                else
                    @test all(counterfactual.x .== counterfactual.f(counterfactual.s′))
                end
                @test converged(counterfactual) == true
    
                # Threshold reached if converged:
                γ = 0.9
                target = round(probs(M, x)[1])==0 ? 1 : 0 
                T = 1000
                counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator; γ=γ, T=T)
                import CounterfactualExplanations.Counterfactuals: counterfactual_probability
                @test !converged(counterfactual) || (target_probs(counterfactual)[1] >= γ) # either not converged or threshold reached
                @test !converged(counterfactual) || length(path(counterfactual)) <= T
    
            end
        end
    
    end
end

