using CSV
using DataFrames
using ghr_jll
using LazyArtifacts
using LibGit2
using MLJBase
using MLJModels: ContinuousEncoder, OneHotEncoder, Standardizer
using MLUtils
using Pkg.Artifacts
using Plots
using Serialization
using StatsBase

# Artifacts:
artifact_toml = LazyArtifacts.find_artifacts_toml(".")

"""
    www_dir(dir="")

Sets up the directory to save images and returns the path.
"""
function www_dir(dir="")
    root_ = "dev/artifacts/upload/www"
    www_dir = joinpath(root_, dir)
    if !isdir(www_dir)
        mkpath(www_dir)
    end
    return www_dir
end

"""
    data_dir(dir="")

Sets up the directory to save data and returns the path.
"""
function data_dir(dir="")
    root_ = "dev/artifacts/upload/data"
    _path = joinpath(root_, dir)
    if !isdir(_path)
        mkpath(_path)
    end
    return _path
end

"""
    model_dir(dir="")

Sets up the directory to save models and returns the path.
"""
function model_dir(dir="")
    root_ = "dev/artifacts/upload/model"
    _path = joinpath(root_, dir)
    if !isdir(_path)
        mkpath(_path)
    end
    return _path
end

function generate_artifacts(
    datafiles;
    artifact_name=nothing,
    root=".",
    artifact_toml=joinpath(root, "Artifacts.toml"),
    deploy=true,
    tag="artifacts-$(Int(VERSION.major)).$(Int(VERSION.minor))",
)
    if deploy && !haskey(ENV, "GITHUB_TOKEN")
        @warn "For automatic github deployment, need GITHUB_TOKEN. Not found in ENV, attemptimg global git config."
    end

    if deploy
        # Where we will put our tarballs
        tempdir = mktempdir()

        # Try to detect where we should upload these weights to (or just override
        # as shown in the commented-out line)
        origin_url = get_git_remote_url(root)
        deploy_repo = "$(basename(dirname(origin_url)))/$(basename(origin_url))"

        # I had to use this hack to get the correct repo name (otherwise I was getting `.git` in the end and it was failing):
        # deploy_repo = "JuliaTrustworthyAI/CounterfactualExplanations.jl"
    end

    # Name for hash/artifact:
    artifact_name = create_artifact_name_from_path(datafiles, artifact_name)

    # create_artifact() returns the content-hash of the artifact directory once we're finished creating it
    hash = create_artifact() do artifact_dir
        cp(datafiles, joinpath(artifact_dir, artifact_name))
    end

    # Spit tarballs to be hosted out to local temporary directory:
    if deploy
        tarball_hash = archive_artifact(hash, joinpath(tempdir, "$(artifact_name).tar.gz"))

        # Calculate tarball url
        tarball_url = "https://github.com/$(deploy_repo)/releases/download/$(tag)/$(artifact_name).tar.gz"

        # Bind this to an Artifacts.toml file
        @info("Binding $(artifact_name) in Artifacts.toml...")
        bind_artifact!(
            artifact_toml,
            artifact_name,
            hash;
            download_info=[(tarball_url, tarball_hash)],
            lazy=true,
            force=true,
        )
    end

    if deploy
        # Upload tarballs to a special github release
        @info("Uploading tarballs to $(deploy_repo) tag `$(tag)`")

        ghr() do ghr_exe
            println(
                readchomp(
                    `$ghr_exe -replace -u $(dirname(deploy_repo)) -r $(basename(deploy_repo)) $(tag) $(tempdir)`,
                ),
            )
        end

        @info("Artifacts.toml file now contains all bound artifact names")
    end
end

function create_artifact_name_from_path(
    datafiles::String, artifact_name::Union{Nothing,String}
)
    # Name for hash/artifact:
    artifact_name =
        isnothing(artifact_name) ? replace(datafiles, ("/" => "-")) : artifact_name
    return artifact_name
end

function get_git_remote_url(repo_path::String)
    repo = LibGit2.GitRepo(repo_path)
    origin = LibGit2.get(LibGit2.GitRemote, repo, "origin")
    return LibGit2.url(origin)
end
