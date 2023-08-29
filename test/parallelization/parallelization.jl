using MPI
using MPIPreferences

nprocs_str = get(ENV, "JULIA_MPI_TEST_NPROCS", "")
nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 8) : parse(Int, nprocs_str)

@testset "MPI" begin
    n = nprocs          # number of processes
    mpiexec() do exe    # MPI wrapper
        run(`$exe -n $n $(Base.julia_cmd()) parallelization/mpi.jl`)
    end
end