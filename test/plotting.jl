using MLUtils
using Plots

M = LogisticModel([1 1],[0])
generator = generator_catalog[:generic]()

@testset "Two-dimensional" begin

    X, ys = toy_data_linear()
    X = MLUtils.stack(X,dims=2)
    
    # Model:
    counterfactual_data = CounterfactualData(X,ys')
    plt = plot(M, counterfactual_data)

    # Counterfactual:
    x = select_factual(counterfactual_data,rand(1:size(X,2)))
    p_ = probs(M, x)
    y = round(p_[1])
    target = y == 0 ? 1 : 0 
    counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator)
    plt = plot(counterfactual)
    anim = animate_path(counterfactual)

end

@testset "Multi-dimensional" begin

    # Model:
    X, ys = toy_data_linear()
    X = MLUtils.stack(X,dims=2)
    X = vcat(X, randn(1,size(X,2)))
    counterfactual_data = CounterfactualData(X,ys')
    M = LogisticModel([1 1 0],[0])
    plt = plot(M, counterfactual_data)
    
end