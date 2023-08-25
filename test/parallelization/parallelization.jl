using MPI

@testset "MPI" begin
    n = 4  # number of processes
    mpiexec() do exe  # MPI wrapper
        run(`$exe -n $n $(Base.julia_cmd()) parallelization/mpi.jl`)
    end
end