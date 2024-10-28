using BenchmarkTools

# Counteractual data and model:
data = CounterfactualData(TaijaData.load_linearly_separable()...)
M = fit_model(data, :Linear)
target = 2
factual = 1
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Search:
generator = GenericGenerator()

if VERSION >= v"1" && !Sys.iswindows()
    t = @benchmark generate_counterfactual(x, target, data, M, generator) samples =
        1000
    expected_allocs = 6000
    @test t.allocs <= expected_allocs
end
