using Flux

"""
    hinge_loss(ŷ, y)

Uses [Flux.Losses.hinge_loss](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.hinge_loss) to compute Hinge loss. If provided labels are in `{0,1}` they are mapped to `{-1,1}`. 
"""
function hinge_loss(ŷ, y)
    h(y) = ifelse(y==0,-1,1) # adapt in case labels are in {0,1}
    return Flux.Losses.hinge_loss(ŷ, z(y))
end