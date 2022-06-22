using RCall, PyCall

# Prepare R
@info "Check for R package dependencies and install if missing."
R"""
deps <- c(
    "torch"
)
lapply(
    deps,
    function(dep) {
        if (!require(dep, character.only=TRUE)) {
            install.packages(dep)
            require(dep, character.only=TRUE)
        }
    }
)
"""
@info "All R package dependencies installed."