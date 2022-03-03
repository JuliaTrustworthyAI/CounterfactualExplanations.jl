module Data

using Pkg.Artifacts
using Flux
using BSON: @load

function ucr_data()

    data_dir = artifact"ucr_data"
    
    @load joinpath(data_dir,"ucr_data.bson") data

    return data

end

end