using Plots
import Plots: scatter!

function scatter!(data::CounterfactualData; kwargs...)
    X, y = unpack(data)
    Plots.scatter!(X[1,:],X[2,:],group=Int.(vec(y)),color=Int.(vec(y)); kwargs...)
end