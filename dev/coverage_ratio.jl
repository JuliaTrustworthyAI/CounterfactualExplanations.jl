using Coverage
using Logging

function coverage_ratio(folder="src")
    with_logger(NullLogger()) do
        # process '*.cov' files
        coverage = process_folder(folder) # defaults to src/; alternatively, supply the folder name as argument
        # Get total coverage for all Julia files
        global covered_lines, total_lines = get_summary(coverage)
    end
    @info "Coverage ratio ($folder): $(round(covered_lines/total_lines*100, digits=2))%"
end

coverage_ratio()
