### A Pluto.jl notebook ###
# v0.19.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try
            Base.loaded_modules[Base.PkgId(
                Base.UUID("6e696c72-6542-2067-7265-42206c756150"),
                "AbstractPlutoDingetjes",
            )].Bonds.initial_value
        catch
            b -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 115c3ad3-3fc3-4d76-bf99-3032d59af96f
begin
    using Pkg
    Pkg.activate("../")
    using CounterfactualExplanations
    using CounterfactualExplanations.Models
    using CounterfactualExplanations.Data
    using Plots
    using PlutoUI
end;

# ╔═╡ ba3373a2-9e41-11ed-3545-cf8d448d384d
md"""
# Whistle-Stop Tour ⏱
"""

# ╔═╡ f3ae9087-8172-4b91-b8d4-25e60f8eff7c
begin
    function multi_slider(vals::Dict; title="")
        return PlutoUI.combine() do Child
            inputs = [
                md""" $(_name): $(
                    Child(_name, Slider(_vals[1], default=_vals[2], show_value=true))
                )"""

                for (_name, _vals) in vals
            ]

            md"""
            #### $title
            $(inputs)
            """
        end
    end
end;

# ╔═╡ b50fe079-be19-4adc-b4da-25a322fd3cde
@bind data_name Select(collect(keys(data_catalogue[:synthetic])))

# ╔═╡ Cell order:
# ╠═ba3373a2-9e41-11ed-3545-cf8d448d384d
# ╠═115c3ad3-3fc3-4d76-bf99-3032d59af96f
# ╟─f3ae9087-8172-4b91-b8d4-25e60f8eff7c
# ╠═b50fe079-be19-4adc-b4da-25a322fd3cde
