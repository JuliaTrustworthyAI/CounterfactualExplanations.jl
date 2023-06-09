using LazyArtifacts

function generate_artifact_dir(name::String)
    _artifacts_julia_version = "$(Int(VERSION.major)).$(Int(VERSION.minor))"
    candidate_name = "$name-$(_artifacts_julia_version)"
    artifact_dir = try
        joinpath(@artifact_str(candidate_name), candidate_name)
    catch
        @warn "Package artifacts have been serialized on version $(_artifacts_julia_version) and may not be compatible with your Julia version."
        joinpath(@artifact_str(name), name)
    end
    return artifact_dir
end
