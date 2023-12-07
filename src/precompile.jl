# using PrecompileTools

# @setup_workload begin
#     y_target = 2
#     y_fact = 1

#     @compile_workload begin

#         # Counterfactual data and model:
#         counterfactual_data = load_linearly_separable()
#         M = fit_model(counterfactual_data, :Linear)
#         chosen = rand(findall(predict_label(M, counterfactual_data) .== y_fact))
#         x = select_factual(counterfactual_data, chosen)

#         # Search:
#         generator = GenericGenerator()
#         ce = generate_counterfactual(x, y_target, counterfactual_data, M, generator)
#     end
# end
