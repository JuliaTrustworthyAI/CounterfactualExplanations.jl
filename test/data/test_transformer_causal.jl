using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing

using Tables

using Tables

using HTTP, CSV, DataFrames
using CausalInference
import CausalInference as CI

using Plots, GraphRecipes, Graphs
# If you have problems with TikzGraphs.jl, 
# try alternatively plotting backend GraphRecipes.jl + Plots.jl
# and corresponding plotting function `plot_pc_graph_recipes`

url = "https://www.ccd.pitt.edu//wp-content/uploads/files/Retention.txt"

df = DataFrame(CSV.File(HTTP.get(url).body))

# for now, pcalg and fcialg only accepts Float variables...
# this should change soon hopefully
for name in names(df)
	df[!, name] = convert(Array{Float64,1}, df[!,name])
end

# make variable names a bit easier to read
variables = map(x->replace(x,"_"=>" "), names(df))

est_g, score = ges(df, penalty=1.0, parallel=true)

est_dag= pdag2dag!(est_g)


display(CI.estimate_equations(df,est_dag))


data= CounterfactualData(Tables.matrix(df),[0,1,1,2,1,1,1,1])

fit_transformer!(data, CI.SCM)
