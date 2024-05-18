using LazyArtifacts: LazyArtifacts, @artifact_str

function generate_artifact_dir(name::String)
    artifact_dir = joinpath(@artifact_str(name), name)
    return artifact_dir
end
