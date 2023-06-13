When running `PythonCall.pyimport("torch")` after `R"library(torch)`:

`ERROR: Python: OSError: [WinError 127] The specified procedure could not be found. Error loading "C:\Users\kicio\CounterfactualExplanations.jl\dev\R_call_implementation\.CondaPkg\env\Lib\site-packages\torch\lib\torch_cpu.dll" or one of its dependencies.`

The other way around:

`ERROR: REvalError: Error in torch_tensor_cpp(data, dtype, device, requires_grad, pin_memory) : Lantern is not loaded. Please use `install_torch()` to install additional dependencies.`


ERROR: LoadError: Some tests did not pass: 24 passed, 0 failed, 1 errored, 0 broken.
in expression starting at C:\Users\kicio\CounterfactualExplanations.jl\test\runtests.jl:22
┌ Error: mktempdir cleanup
│   exception =
│    IOError: rm("C:\\Users\\kicio\\AppData\\Local\\Temp\\jl_927dif\\.CondaPkg\\env\\DLLs"): directory not empty (ENOTEMPTY)
│    Stacktrace:
│      [1] uv_error
│        @ .\libuv.jl:97 [inlined]
│      [2] rm(path::String; force::Bool, recursive::Bool)
│        @ Base.Filesystem .\file.jl:306
│      [3] rm(path::String; force::Bool, recursive::Bool) (repeats 3 times)
│        @ Base.Filesystem .\file.jl:294
│      [4] mktempdir(fn::Pkg.Operations.var"#105#110"{Dict{String, Any}, Bool, Bool, Bool, Pkg.Operations.var"#120#125"{Bool, Cmd, Cm)
│        @ Base.Filesystem .\file.jl:769
│      [5] mktempdir(fn::Function, parent::String) (repeats 2 times)
│        @ Base.Filesystem .\file.jl:760
│      [6] sandbox(fn::Function, ctx::Pkg.Types.Context, target::Pkg.Types.PackageSpec, target_path::String, sandbox_path::String, sa)
│        @ Pkg.Operations C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\Operations.jl:1540        
│      [7] test(ctx::Pkg.Types.Context, pkgs::Vector{Pkg.Types.PackageSpec}; coverage::Bool, julia_args::Cmd, test_args::Cmd, test_fn)
│        @ Pkg.Operations C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\Operations.jl:1746        
│      [8] test(ctx::Pkg.Types.Context, pkgs::Vector{Pkg.Types.PackageSpec}; coverage::Bool, test_fn::Nothing, julia_args::Cmd, test_)
│        @ Pkg.API C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\API.jl:434
│      [9] test(pkgs::Vector{Pkg.Types.PackageSpec}; io::Base.TTY, kwargs::Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{(), Tuple{)
│        @ Pkg.API C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\API.jl:156
│     [10] test(pkgs::Vector{Pkg.Types.PackageSpec})
│        @ Pkg.API C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\API.jl:145
│     [11] do_cmd!(command::Pkg.REPLMode.Command, repl::REPL.LineEditREPL)
│        @ Pkg.REPLMode C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\REPLMode\REPLMode.jl:409    
│     [12] do_cmd(repl::REPL.LineEditREPL, input::String; do_rethrow::Bool)
│        @ Pkg.REPLMode C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\REPLMode\REPLMode.jl:387    
│     [13] do_cmd
│        @ C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\REPLMode\REPLMode.jl:377 [inlined]       
│     [14] (::Pkg.REPLMode.var"#24#27"{REPL.LineEditREPL, REPL.LineEdit.Prompt})(s::REPL.LineEdit.MIState, buf::IOBuffer, ok::Bool)   
│        @ Pkg.REPLMode C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\Pkg\src\REPLMode\REPLMode.jl:551    
│     [15] #invokelatest#2
│        @ .\essentials.jl:729 [inlined]
│     [16] invokelatest
│        @ .\essentials.jl:726 [inlined]
│     [17] run_interface(terminal::REPL.Terminals.TextTerminal, m::REPL.LineEdit.ModalInterface, s::REPL.LineEdit.MIState)
│        @ REPL.LineEdit C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\REPL\src\LineEdit.jl:2510
│     [18] run_frontend(repl::REPL.LineEditREPL, backend::REPL.REPLBackendRef)
│        @ REPL C:\Users\kicio\AppData\Local\Programs\Julia-1.8.5\share\julia\stdlib\v1.8\REPL\src\REPL.jl:1248
│     [19] (::REPL.var"#49#54"{REPL.LineEditREPL, REPL.REPLBackendRef})()
│        @ REPL .\task.jl:484
└ @ Base.Filesystem file.jl:772
ERROR: Package CounterfactualExplanations errored during testing