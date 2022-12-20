using CounterfactualExplanations
using CounterfactualExplanations.Benchmark
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Data
using CounterfactualExplanations.Models
using DataFrames
using Flux
using LinearAlgebra
using MLUtils
using Random
init_perturbation = 2.0

# NOTE:
# This is probably the most important/useful test script, because it runs through the whole process of: 
# - loading artifacts
# - setting up counterfactual search for various models and generators
# - running counterfactual search

# LOOP:
for (key, generator_) ∈ generators
    name = uppercasefirst(string(key))
    @testset "$name" begin

        # Generator:
        generator = deepcopy(generator_())

        @testset "Models for synthetic data" begin

            for (key, value) ∈ synthetic

                name = string(key)
                @testset "$name" begin
                    X, ys = (value[:data][:xs], value[:data][:ys])
                    X = MLUtils.stack(X, dims = 2)
                    ys_cold =
                        length(ys[1]) > 1 ?
                        [Flux.onecold(y_, 1:length(ys[1])) for y_ in ys] : ys
                    counterfactual_data = CounterfactualData(X, ys')

                    for (likelihood, model) ∈ value[:models]
                        name = string(likelihood)
                        @testset "$name" begin
                            M = model[:model]
                            # Randomly selected factual:
                            Random.seed!(123)
                            x = select_factual(counterfactual_data, rand(1:size(X)[2]))
                            multiple_x =
                                select_factual(counterfactual_data, rand(1:size(X)[2], 5))

                            p_ = probs(M, x)
                            if size(p_)[1] > 1
                                y = Flux.onecold(p_, unique(ys_cold))
                                target = rand(unique(ys_cold)[1:end.!=y]) # opposite label as target
                            else
                                y = round(p_[1])
                                target = y == 0 ? 1 : 0
                            end
                            # Single sample:
                            counterfactual = generate_counterfactual(
                                x,
                                target,
                                counterfactual_data,
                                M,
                                generator,
                            )
                            # Multiple samples:
                            counterfactuals = generate_counterfactual(
                                multiple_x,
                                target,
                                counterfactual_data,
                                M,
                                generator,
                            )

                            @testset "Predetermined outputs" begin
                                if typeof(generator) <:
                                   Generators.AbstractLatentSpaceGenerator
                                    @test counterfactual.latent_space
                                end
                                @test counterfactual.target == target
                                @test counterfactual.x == x &&
                                      CounterfactualExplanations.factual(counterfactual) ==
                                      x
                                @test CounterfactualExplanations.factual_label(
                                    counterfactual,
                                ) == y
                                @test CounterfactualExplanations.factual_probability(
                                    counterfactual,
                                ) == p_
                            end

                            @testset "Benchmark" begin
                                @test isa(benchmark(counterfactual), DataFrame)
                                @test isa(
                                    benchmark(counterfactuals; to_dataframe = false),
                                    Dict,
                                )
                            end

                            @testset "Convergence" begin

                                @testset "Non-trivial case" begin
                                    counterfactual_data.generative_model = nothing
                                    # Threshold reached if converged:
                                    γ = 0.9
                                    generator.decision_threshold = γ
                                    T = 1000
                                    counterfactual = generate_counterfactual(
                                        x,
                                        target,
                                        counterfactual_data,
                                        M,
                                        generator;
                                        T = T,
                                    )
                                    using CounterfactualExplanations:
                                        counterfactual_probability
                                    @test !converged(counterfactual) ||
                                          target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                                    @test !converged(counterfactual) ||
                                          length(path(counterfactual)) <= T
                                end

                                @testset "Trivial case (already in target class)" begin
                                    counterfactual_data.generative_model = nothing
                                    # Already in target and exceeding threshold probability:
                                    p_ = probs(M, x)
                                    if size(p_)[1] > 1
                                        y = Flux.onecold(p_, unique(ys_cold))[1]
                                        target = y
                                    else
                                        target = round(p_[1]) == 0 ? 0 : 1
                                    end
                                    generator.decision_threshold = 0.5
                                    counterfactual = generate_counterfactual(
                                        x,
                                        target,
                                        counterfactual_data,
                                        M,
                                        generator;
                                    )
                                    @test length(path(counterfactual)) == 1
                                    @test maximum(
                                        abs.(
                                            counterfactual.x .-
                                            CounterfactualExplanations.decode_state(
                                                counterfactual,
                                            )
                                        ),
                                    ) < init_perturbation
                                    @test converged(counterfactual)
                                    @test CounterfactualExplanations.terminated(
                                        counterfactual,
                                    )
                                    @test CounterfactualExplanations.total_steps(
                                        counterfactual,
                                    ) == 0

                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
