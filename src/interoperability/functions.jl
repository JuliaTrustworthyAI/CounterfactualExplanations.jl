using RCall, PyCall

struct InteropError <: Exception 
    lang::String
end

Base.showerror(io::IO, e::InteropError) = print(io, typeof(e), ": One or multiple $(e.lang) package dependencies missing. Automatic installation failed.")

#################################
############### R ###############
#################################
function prep_R_session()
    try
        R"""
        deps <- c(
            "torch"
        )
        lapply(
            deps,
            function(dep) {
                if (!require(dep, character.only=TRUE, quietly=TRUE)) {
                    message("Installing missing R package dependencies.")
                    install.packages(dep)
                    require(dep, character.only=TRUE, quietly=TRUE)
                    message("All R package dependencies installed.")
                }
            }
        )
        """
    catch
        throw(InteropError("R"))
    end
end