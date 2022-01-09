module Convergence

function zero_gradient
end

function confidence(x̅::Vector{x}, 𝓜, target::Float64; τ)
    𝐠ₜ = gradient(() -> loss(x, y), params(W, b))
end

end