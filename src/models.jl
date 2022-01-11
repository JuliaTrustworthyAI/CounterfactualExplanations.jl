# --------------- Base type for model:
abstract type Model end

# -------- Linear model:
abstract type LinearModel <: Model end
predict(ℳ::LinearModel, x, w) = w'x
