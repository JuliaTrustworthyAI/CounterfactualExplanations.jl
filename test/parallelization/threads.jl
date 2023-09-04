using CounterfactualExplanations
using CounterfactualExplanations.Parallelization

counterfactual_data = synthetic[:classification_binary][:data]
M = synthetic[:classification_binary][:models][:MLP][:model]
generator = GenericGenerator()
factual = 1
target = 2
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual), 1000)
xs = select_factual(counterfactual_data, chosen)

parallelizer = ThreadsParallelizer()
ces = @with_parallelizer parallelizer begin
    generate_counterfactual(xs, target, counterfactual_data, M, generator)
end

@test true
