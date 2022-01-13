"""
    hinge_loss(ŷ, y)

Uses [Flux.hinge_loss](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.hinge_loss) to compute Hinge loss. If provided labels are in `\{0,1\}` they are mapped to `\{-1,1\}`. 
"""
function logitbinarycrossentropy(ŷ, y)
    z(y) = ifelse(y==0,-1,1) # adapt in case labels are in {0,1}
    return Flux.logitbinarycrossentropy(z(ŷ), z(y))
end