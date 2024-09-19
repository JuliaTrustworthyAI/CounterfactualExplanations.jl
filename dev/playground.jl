using CounterfactualExplanations
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using DecisionTree
using Plots
using TaijaData
using TaijaPlotting

# Counteractual data and model:
n = 3000
data = CounterfactualData(load_moons(n; noise=0.25)...)
X = data.X
M = fit_model(data, :MLP)
fx = predict_label(M, data)
target = 1
factual = 0
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Generic search:
generator = Generators.GenericGenerator()
ce = generate_counterfactual(x, target, data, M, generator)

# T-CREx ###################################################################
ρ = 0.02
τ = 0.9
generator = Generators.TCRExGenerator(ρ)

# (a) ##############################

# Surrogate:
model, fitresult = Generators.grow_surrogate(generator, ce)
M_sur = CounterfactualExplanations.DecisionTreeModel(model; fitresult=fitresult)
plot(M_sur, data; ms=3, markerstrokewidth=0, size=(500, 500), colorbar=false)

# Extract rules:
R = Generators.extract_rules(fitresult[1])
print_tree(fitresult[1])

# Compute feasibility and accuracy:
feas = Generators.rule_feasibility.(R, (X,))
@assert minimum(feas) >= ρ
acc_factual = Generators.rule_accuracy.(R, (X,), (fx,), (factual,))
acc_target = Generators.rule_accuracy.(R, (X,), (fx,), (target,))
@assert all(acc_target .+ acc_factual .== 1.0)

# (b) ##############################
R_max = Generators.max_valid(R, X, fx, target, τ)
feas_max = Generators.rule_feasibility.(R_max, (X,))
acc_max = Generators.rule_accuracy.(R_max, (X,), (fx,), (target,))
plt = plot(data; ms=3, markerstrokewidth=0, size=(500, 500))
rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
for (i, rule) in enumerate(R_max)
    ubx, uby = minimum([rule[1][2], maximum(X[1, :])]),
    minimum([rule[2][2], maximum(X[2, :])])
    lbx, lby = maximum([rule[1][1], minimum(X[1, :])]),
    maximum([rule[2][1], minimum(X[2, :])])
    _feas = round(feas_max[i]; digits=2)
    _n = Int(round(feas_max[i] * n; digits=2))
    _acc = round(acc_max[i]; digits=2)
    @info "Rectangle R$i with feasibility $(_feas) (n≈$(_n)) and accuracy $(_acc)"
    lab = "R$i (ρ̂=$(_feas), τ̂=$(_acc))"
    plot!(plt, rectangle(ubx-lbx,uby-lby,lbx,lby), opacity=.5, color=i+2, label=lab)
end
plt

# (c) ##############################
lbsx, ubsx = Generators.partition_bounds(R_max, 1)
lbsy, ubsy = Generators.partition_bounds(R_max, 2)

for (i, (lbx, ubx)) in enumerate(zip(lbsx, ubsx))
    @info "X bounds $i"
    for (j, (lby, uby)) in enumerate(zip(lbsy, ubsy))
        @info "Y bounds $j"
        lbx = maximum([lbx, minimum(X[1, :])])
        lby = maximum([lby, minimum(X[2, :])])
        ubx = minimum([ubx, maximum(X[1, :])])
        uby = minimum([uby, maximum(X[2, :])])
        idx = (i - 1) * length(zip(lbsy, ubsy)) + j
        plot!(
            plt,
            rectangle(ubx - lbx, uby - lby, lbx, lby);
            fillcolor="white",
            fillalpha=0.0,
            label=nothing,
            lw=2,
        )
    end
end
plt