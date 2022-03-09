module Data

using Pkg.Artifacts
using Flux
using BSON: @load

# UCR data:
function ucr_data()
    data_dir = artifact"ucr_data"
    @load joinpath(data_dir,"ucr_data.bson") data
    return data
end

function ucr_model()
    data_dir = artifact"ucr_model"
    @load joinpath(data_dir,"ucr_model.bson") model
    return model
end

# Cats and dogs:
function cats_dogs()
    data_dir = artifact"cats_dogs_data"
    @load joinpath(data_dir,"cats_dogs_data.bson") data
    return data
end

end