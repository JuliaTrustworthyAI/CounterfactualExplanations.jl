module Data

using Pkg.Artifacts
using Flux
using BSON

# UCR data:
function ucr_data()
    data_dir = artifact"ucr_data"
    data = BSON.load(joinpath(data_dir,"ucr_data.bson"),@__MODULE__)
    return data
end

function ucr_model()
    data_dir = artifact"ucr_model"
    model = BSON.load(joinpath(data_dir,"ucr_model.bson"),@__MODULE__)
    return model
end

# Cats and dogs:
function cats_dogs()
    data_dir = artifact"cats_dogs_data"
    data = BSON.load(joinpath(data_dir,"cats_dogs_data.bson"),@__MODULE__)
    return data
end

end