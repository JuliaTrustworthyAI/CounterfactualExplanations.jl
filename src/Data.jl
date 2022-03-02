module Data

using Pkg.Artifacts
using Flux
using BSON: @load

function ucr_data()

    data_dir = artifact"UCR"
    
    @load joinpath(data_dir,"data.bson") data
    @load joinpath(data_dir,"model.bson") model

    return data, model

end

end