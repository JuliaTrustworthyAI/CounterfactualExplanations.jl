# Define a basic node structure for the decision tree
struct TreeNode{S,T}
    feature::Int           # The feature index to split on
    threshold::S     # The threshold to split on
    left::Union{TreeNode{S,T},Nothing}   # Left child (if any)
    right::Union{TreeNode{S,T},Nothing}  # Right child (if any)
    prediction::Union{Nothing,T}  # Class label for leaf node
    values::Union{Nothing,Vector{T}}      # The sample labels for leaf node
end

function wrap_decision_tree end

# Function to check if a node is a leaf node
is_leaf(node::TreeNode) = !isnothing(node.prediction)

# Helper function
function countmap(y)
    counts = Dict{Int,Int}()
    for label in y
        counts[label] = get(counts, label, 0) + 1
    end
    return counts
end

# Function to calculate Gini impurity (or any other impurity metric)
function gini_impurity(y)
    classes, counts = unique(y), countmap(y)
    total = length(y)
    return 1.0 - sum((counts[class] / total)^2 for class in classes)
end

# Function to split data based on a feature and a threshold
function split_data(X, y, feature, threshold)
    # Find indices where the feature value is less than or equal to the threshold
    left_indices = findall(x -> x[feature] <= threshold, eachrow(X))
    right_indices = findall(x -> x[feature] > threshold, eachrow(X))

    # Extract corresponding rows from X and y based on the indices
    left_X = X[left_indices, :]
    left_y = y[left_indices]
    right_X = X[right_indices, :]
    right_y = y[right_indices]

    return left_X, left_y, right_X, right_y
end

# Function to build the decision tree recursively
function _build_tree(X, y, max_depth, current_depth, allowed_thresholds)
    allowed_thresholds = [
        bounds[.!isinf.(bounds)] |> x -> convert.(eltype(X), x) for
        bounds in allowed_thresholds
    ]

    # If all samples are of the same class, create a leaf node
    if length(unique(y)) == 1 || current_depth == max_depth
        return TreeNode(-1, convert(eltype(X), -1.0), nothing, nothing, unique(y)[1], y)
    end

    # Track the best feature and threshold for splitting
    best_gini, best_feature, best_threshold = Inf, -1, -1
    best_left_X, best_left_y, best_right_X, best_right_y = nothing,
    nothing, nothing,
    nothing

    # Loop through each feature and each allowed threshold for that feature
    for feature in 1:size(X, 2)
        for threshold in allowed_thresholds[feature]
            left_X, left_y, right_X, right_y = split_data(X, y, feature, threshold)
            if isempty(left_y) || isempty(right_y)
                continue
            end

            # Calculate the weighted Gini impurity for the split
            gini_left = gini_impurity(left_y)
            gini_right = gini_impurity(right_y)
            weighted_gini =
                (length(left_y) / length(y)) * gini_left +
                (length(right_y) / length(y)) * gini_right

            # Update the best split if the current one is better
            if weighted_gini < best_gini
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold
                best_left_X, best_left_y = left_X, left_y
                best_right_X, best_right_y = right_X, right_y
            end
        end
    end

    # If no valid split was found, return a leaf node
    if best_feature == -1
        return TreeNode(-1, convert(eltype(X), -1.0), nothing, nothing, unique(y)[1], y)
    end

    # Recursively build the left and right subtrees
    left_node = _build_tree(
        best_left_X, best_left_y, max_depth, current_depth + 1, allowed_thresholds
    )
    right_node = _build_tree(
        best_right_X, best_right_y, max_depth, current_depth + 1, allowed_thresholds
    )

    # Return the current node (internal node)
    return TreeNode(best_feature, best_threshold, left_node, right_node, nothing, nothing)
end

# Function to predict a single instance using the decision tree
function predict_tree(node::TreeNode, x::AbstractVector)
    if is_leaf(node)
        return node.prediction
    elseif x[node.feature] <= node.threshold
        return predict_tree(node.left, x)
    else
        return predict_tree(node.right, x)
    end
end

# Function to predict multiple instances using the decision tree
function predict_tree(node::TreeNode, X::AbstractMatrix)
    return [predict_tree(node, x) for x in eachrow(X)]
end

# # Sample usage:

# # Define the input data (X) and labels (y)
# X = [2.5 1.0; 1.5 2.0; 3.0 2.5; 0.5 1.5]
# y = [1, 0, 1, 0]

# # Define allowed thresholds for each feature
# allowed_thresholds = [[0.5, 1.5, 2.5], [1.0, 1.5, 2.0]]

# # Build the decision tree
# max_depth = 3
# tree = _build_tree(X, y, max_depth, 0, allowed_thresholds)

# # Predict using the trained tree
# predictions = predict_tree(tree, X)
# println("Predictions: ", predictions)
