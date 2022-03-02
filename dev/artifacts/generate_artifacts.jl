using Pkg.Artifacts

# This is the path to the Artifacts.toml we will manipulate
artifact_toml = joinpath(@__DIR__, "../..", "Artifacts.toml")

data_repo = "https://github.com/pat-alt/AlgorithmicRecourse_data/raw/main"

function generate_artifact(name; data_repo=data_repo, artifact_toml=artifact_toml)

    hash = artifact_hash(name, artifact_toml)

    # If the name was not bound, or the hash it was bound to does not exist, create it!
    if isnothing(hash) || !artifact_exists(hash)

        # We create the artifact by simply downloading a few files into the new artifact directory
        url_base = joinpath(data_repo,name)

        # create_artifact() returns the content-hash of the artifact directory once we're finished creating it
        hash = create_artifact() do artifact_dir
            download("$(url_base)/data.bson", joinpath(artifact_dir, "data.bson"))
            download("$(url_base)/model.bson", joinpath(artifact_dir, "model.bson"))
        end

        # Now bind that hash within our `Artifacts.toml`.  `force = true` means that if it already exists,
        # just overwrite with the new content-hash.  Unless the source files change, we do not expect
        # the content hash to change, so this should not cause unnecessary version control churn.
        bind_artifact!(artifact_toml, name, hash; lazy=true)
    end
end

generate_artifact("UCR")
