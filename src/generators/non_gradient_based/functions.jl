"""
    search_path(estimator, class_labels, target)

Return a path index list with the ids of the leaf nodes, inequality symbols, thresholds and feature indices
"""
function search_path(decision_tree, class_labels, target)
    children_left = decision_tree[:tree_][:children_left]
    children_right = decision_tree[:tree_][:children_right]
    feature = decision_tree[:tree_][:feature]
    threshold = decision_tree[:tree_][:threshold]
    leaf_nodes = findall(children_left .== -1)
    leaf_values = reshape(decision_tree[:tree_][:value][leaf_nodes], length(leaf_nodes), length(class_labels))
    leaf_nodes = findall(leaf_values[:, target] .!= 0)

    paths = Dict()
    for leaf_node in leaf_nodes
        child_node = leaf_node
        parent_node = -100
        parents_left = []
        parents_right = []
        while parent_node != 0
            if length(findall(children_left .== child_node)) == 0
                parent_left = -1
                parent_right = findfirst(children_right .== child_node)
                parent_node = parent_right
            elseif length(findall(children_right .== child_node)) == 0
                parent_right = -1
                parent_left = findfirst(children_left .== child_node)
                parent_node = parent_left
            end
            push!(parents_left, parent_left)
            push!(parents_right, parent_right)
            child_node = parent_node
        end
        paths[leaf_node] = (parents_left, parents_right)
    end

    path_info = Dict()
    for i in keys(paths)
        node_ids = []
        inequality_symbols = []
        thresholds = []
        features = []
        parents_left, parents_right = paths[i]
        for idx in 1:length(parents_left)
            if parents_left[idx] != -1
                node_id = parents_left[idx]
                push!(node_ids, node_id)
                push!(inequality_symbols, 0)
                push!(thresholds, threshold[node_id])
                push!(features, feature[node_id])
            elseif parents_right[idx] != -1
                node_id = parents_right[idx]
                push!(node_ids, node_id)
                push!(inequality_symbols, 1)
                push!(thresholds, threshold[node_id])
                push!(features, feature[node_id])
            end
            path_info[i] = Dict("node_id" => node_ids, "inequality_symbol" => inequality_symbols,
                                "threshold" => thresholds, "feature" => features)
        end
    end
    return path_info
end

"""
    feature_tweaking(ensemble::FluxEnsemble, input_data::CounterfactualData, x, class_labels, target, epsilon, loss)

Returns a counterfactual instance of `x` based on the ensemble of classifiers provided.
"""
function feature_tweaking(ensemble::FluxEnsemble, input_data::CounterfactualData, x, class_labels, target, epsilon, loss)
    x_out = deepcopy(x)
    delta = 10^3
    for classifier in ensemble
        if predict_label(ensemble, input_data, x) == predict_label(classifier, input_data, x) &&
            predict_label(classifier, input_data, x) != target
            paths = search_path(classifier, class_labels, target)
            for key in keys(paths)
                path = paths[key]
                es_instance = esatisfactory_instance(x, epsilon, path)
                if predict_label(classifier, input_data, es_instance) == target
                    if loss(x, es_instance) < delta
                        x_out = es_instance
                        delta = loss(x, es_instance)
                    end
                end
            end
        end
    end
    return x_out
end

"""
    esatisfactory_instance(x, epsilon, paths)

Returns an epsilon-satisfactory instance of `x` based on the paths provided.
"""
function esatisfactory_instance(generator::FeatureTweakGenerator, x, epsilon, paths)
    esatisfactory = deepcopy(x)
    for i in 1:length(paths["feature"])
        feature_idx = paths["feature"][i]
        threshold_value = paths["threshold"][i]
        inequality_symbol = paths["inequality_symbol"][i]
        if inequality_symbol == 0
            esatisfactory[feature_idx] = threshold_value - epsilon
        elseif inequality_symbol == 1
            esatisfactory[feature_idx] = threshold_value + epsilon
        else
            println("something wrong")
        end
    end
    return esatisfactory
end