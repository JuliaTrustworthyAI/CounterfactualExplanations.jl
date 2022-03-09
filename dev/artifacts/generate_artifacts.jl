using Pkg.Artifacts, LibGit2, ghr_jll

function generate_artifact(
    datafiles; 
    data_dir="../data", 
    root="../..",
    artifact_toml=joinpath("../..", "Artifacts.toml"), 
    deploy=true,
    tag="data"
)

    if deploy && !haskey(ENV, "GITHUB_TOKEN")
        @warn "For automatic github deployment, need GITHUB_TOKEN. Not found in ENV, attemptimg global git config."
    end

    if deploy
        # Where we will put our tarballs
        tempdir = mktempdir()
    
        function get_git_remote_url(repo_path::String)
            repo = LibGit2.GitRepo(repo_path)
            origin = LibGit2.get(LibGit2.GitRemote, repo, "origin")
            return LibGit2.url(origin)
        end
    
        # Try to detect where we should upload these weights to (or just override
        # as shown in the commented-out line)
        origin_url = get_git_remote_url(root)
        deploy_repo = "$(basename(dirname(origin_url)))/$(splitext(basename(origin_url))[1])"
    
    end

    # Collect all BSON files:
    # datafiles = filter(x->endswith(x,".bson"), readdir(data_dir))

    # For each BSON file, generate its own artifact:
    for datafile in datafiles

        # Name for hash/artifact:
        name = splitext(datafile)[1]

        # create_artifact() returns the content-hash of the artifact directory once we're finished creating it
        hash = create_artifact() do artifact_dir
            cp(joinpath(data_dir, datafile), joinpath(artifact_dir, datafile))
        end

        # Spit tarballs to be hosted out to local temporary directory:
        if deploy
            
            tarball_hash = archive_artifact(hash, joinpath(tempdir, "$(name).tar.gz"))

            # Calculate tarball url
            tarball_url = "https://github.com/$(deploy_repo)/releases/download/$(tag)/$(name).tar.gz"

            # Bind this to an Artifacts.toml file
            @info("Binding $(name) in Artifacts.toml...")
            bind_artifact!(
                artifact_toml, name, hash; 
                download_info=[(tarball_url, tarball_hash)], lazy=true, force=true
            )
        end

    end

    if deploy
        # Upload tarballs to a special github release
        @info("Uploading tarballs to $(deploy_repo) tag `$(tag)`")

        ghr() do ghr_exe
            println(readchomp(`$ghr_exe -replace -u $(dirname(deploy_repo)) -r $(basename(deploy_repo)) $(tag) $(tempdir)`))
        end

        @info("Artifacts.toml file now contains all bound artifact names")
    end

end

# generate_artifact()
