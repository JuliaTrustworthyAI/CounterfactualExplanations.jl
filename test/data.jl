ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "Construction" begin
    for (name, loader) in merge(values(data_catalogue)...)
        @testset "$name" begin
            @test typeof(loader()) == CounterfactualData
        end
    end
end
