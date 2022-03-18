# Plot data points:
using Plots
"""
    plot_data!(plt,X,y)

# Examples

```julia-repl
using CounterfactualExplanations
using CounterfactualExplanations.Data: toy_data_linear
using CounterfactualExplanations.Utils: plot_data!
X, y = toy_data_linear(100)
plt = plot()
plot_data!(plt, hcat(X...)', y)
```

"""
function plot_data!(plt,X,y)
    Plots.scatter!(plt, X[:,1],X[:,2],group=Int.(y),color=Int.(y))
end

# Plot contour of posterior predictive:
using Plots, CounterfactualExplanations.Models
"""
    plot_contour(X,y,𝑴;clegend=true,title="",length_out=50,type=:laplace,zoom=0,xlim=nothing,ylim=nothing)

Generates a contour plot for the posterior predictive surface.  

# Examples

```julia-repl
using CounterfactualExplanations
using CounterfactualExplanations.Data: toy_data_linear
using CounterfactualExplanations.Utils: plot_contour
X, y = toy_data_linear(100)
X = hcat(X...)'
β = [1,1]
𝑴 =(β=β,)
predict(𝑴, X) = σ.(𝑴.β' * X)
plot_contour(X, y, 𝑴)
```

"""
function plot_contour(X,y,𝑴;clegend=true,title="",length_out=50,zoom=-1,xlim=nothing,ylim=nothing,linewidth=0.1)
    
    # Surface range:
    if isnothing(xlim)
        xlim = (minimum(X[:,1]),maximum(X[:,1])).+(zoom,-zoom)
    else
        xlim = xlim .+ (zoom,-zoom)
    end
    if isnothing(ylim)
        ylim = (minimum(X[:,2]),maximum(X[:,2])).+(zoom,-zoom)
    else
        ylim = ylim .+ (zoom,-zoom)
    end
    x_range = collect(range(xlim[1],stop=xlim[2],length=length_out))
    y_range = collect(range(ylim[1],stop=ylim[2],length=length_out))
    Z = [Models.probs(𝑴,[x, y])[1] for x=x_range, y=y_range]

    # Plot:
    plt = contourf(
        x_range, y_range, Z'; 
        colorbar=clegend, title=title, linewidth=linewidth,
        xlim=xlim,
        ylim=ylim
    )
    plot_data!(plt,X,y)

end

# Plot contour of posterior predictive:
using Plots, CounterfactualExplanations.Models
"""
    plot_contour_multi(X,y,𝑴;clegend=true,title="",length_out=50,type=:laplace,zoom=0,xlim=nothing,ylim=nothing)

Generates a contour plot for the posterior predictive surface.  

# Examples

```julia-repl
using BayesLaplace, Plots
import BayesLaplace: predict
using NNlib: σ
X, y = toy_data_linear(100)
X = hcat(X...)'
β = [1,1]
𝑴 =(β=β,)
predict(𝑴, X) = σ.(𝑴.β' * X)
plot_contour(X, y, 𝑴)
```

"""
function plot_contour_multi(X,y,𝑴;
    target::Union{Nothing,Number}=nothing,title="",length_out=50,zoom=-1,xlim=nothing,ylim=nothing,linewidth=0.1)
    
    # Surface range:
    if isnothing(xlim)
        xlim = (minimum(X[:,1]),maximum(X[:,1])).+(zoom,-zoom)
    else
        xlim = xlim .+ (zoom,-zoom)
    end
    if isnothing(ylim)
        ylim = (minimum(X[:,2]),maximum(X[:,2])).+(zoom,-zoom)
    else
        ylim = ylim .+ (zoom,-zoom)
    end
    x_range = collect(range(xlim[1],stop=xlim[2],length=length_out))
    y_range = collect(range(ylim[1],stop=ylim[2],length=length_out))
    Z = reduce(hcat, [Models.probs(𝑴,[x, y]) for x=x_range, y=y_range])

    # Plot:
    if isnothing(target)
        # Plot all contours as lines:
        plt = plot()
        plot_data!(plt,X,y)
        out_dim = size(Z)[1]
        for d in 1:out_dim
            contour!(
                plt,
                x_range, y_range, Z[d,:]; 
                colorbar=false, title=title,
                xlim=xlim,
                ylim=ylim,
                colour=d
            )
        end
    else
        # Print contour fill of target class:
        plt = contourf(
            x_range, y_range, Z[Int(target),:]; 
            colorbar=true, title=title, linewidth=linewidth,
            xlim=xlim,
            ylim=ylim
        )
        plot_data!(plt,X,y)
    end

    return plot(plt)
    
end