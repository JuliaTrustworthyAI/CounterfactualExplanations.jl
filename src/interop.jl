using RCall, PyCall

# Prepare R
@info "Check for R package dependencies and install if missing."
R"""
dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)  # create personal library
.libPaths(Sys.getenv("R_LIBS_USER"))
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