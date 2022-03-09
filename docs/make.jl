using CLEAR
using Documenter

DocMeta.setdocmeta!(CLEAR, :DocTestSetup, :(using CLEAR); recursive=true)

makedocs(;
    modules=[CLEAR],
    authors="Patrick Altmeyer",
    repo="https://github.com/pat-alt/CLEAR.jl/blob/{commit}{path}#{line}",
    sitename="CLEAR.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pat-alt.github.io/CLEAR.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "install.md",
        "Tutorials" =>
            [
                "Binary target" => "tutorials/binary.md",
                "Models" => "tutorials/models.md",
                "Multi-class target" => "tutorials/multi.md",
                "Loss functions" => "tutorials/loss.md"
            ],
        # "Examples" =>
        #     [
        #         "Image data" => [
        #             "MNIST" => "examples/image/MNIST.md"
        #         ],
        #         "Time series" => [
        #             "UCR ECG200" => "examples/timeseries/UCR_ECG200.md"
        #         ]
                
        #     ],
        "Reference" => "reference.md"
    ],
)

deploydocs(;
    repo="github.com/pat-alt/CLEAR.jl",
    devbranch="main"
)
