using CounterfactualExplanations
using CounterfactualExplanations.Models
using CounterfactualExplanations.Counterfactuals
using CounterfactualExplanations.Generators
using Random, LinearAlgebra
Random.seed!(1234)

# Set up:
generators = Dict(
    :generic => Generators.GenericGenerator,
    :greedy => Generators.GreedyGenerator,
    :revise => Generators.REVISEGenerator
)

# Data:
using CounterfactualExplanations.Data
using Random
Random.seed!(1234)
N = 25
xs, ys = Data.toy_data_linear(N)
X = hcat(xs...)
counterfactual_data = CounterfactualData(X,ys')

for (key, generator_) ∈ generators
    name = uppercasefirst(string(key))
    @testset "$name" begin
    
        # Generator:
        generator = generator_()
    
        @testset "LogisticModel" begin
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
                @test counterfactual.x == counterfactual.f(counterfactual.s′)
                @test converged(counterfactual) == true
    
                # Threshold reached if converged:
                γ = 0.9
                target = round(probs(M, x)[1])==0 ? 1 : 0 
                T = 1000
                counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator; γ=γ, T=T)
                import CounterfactualExplanations.Counterfactuals: counterfactual_probability
                @test !converged(counterfactual) || counterfactual_probability(counterfactual)[1] >= γ # either not converged or threshold reached
                @test !converged(counterfactual) || length(path(counterfactual)) <= T
    
            end
        end
    
    end
end

