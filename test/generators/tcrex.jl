@testset "T-CREx" begin
    n = 3000
    data = CounterfactualData(load_moons(n; noise=0.25)...)
    X = data.X
    M = fit_model(data, :MLP)
    fx = predict_label(M, data)
    target = 1
    factual = 0
    chosen = rand(findall(predict_label(M, data) .== factual))
    x = select_factual(data, chosen)

    ρ = 0.02        # feasibility threshold (see Bewley et al. (2024))
    τ = 0.9         # accuracy threshold (see Bewley et al. (2024))
    generator = Generators.TCRExGenerator(; ρ=ρ, τ=τ)
    cre = generator(target, data, M)        # counterfactual rule explanation (global)
    idx, optimal_rule = cre(x)              # counterfactual point explanation (local)
    @test true
end
