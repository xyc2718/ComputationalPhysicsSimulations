
module cellmin

using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using LsqFit
using ..Model
    
export birch_murnaghan,BMfit
"""
BM 函数
:param V: 体积
:param p: 参数=[V0, B0, B0', E0]
"""
function birch_murnaghan(V::Vector{Float64}, p::Vector{Float64})
    V0, B0, B0_prime, E0 = p
    eta = (V0 ./ V).^(2/3)  # 使用广播运算符 ./
    E = E0 .+ (9 * V0 * B0 / 16) .* ((eta .- 1).^3 .* B0_prime .+ (eta .- 1).^2 .* (6 .- 4 .* eta))
    return E
end

function birch_murnaghan(V::Float64, p::Vector{Float64})
    V0, B0, B0_prime, E0 = p
    eta = (V0 ./ V).^(2/3)  # 使用广播运算符 ./
    E = E0 .+ (9 * V0 * B0 / 16) .* ((eta .- 1).^3 .* B0_prime .+ (eta .- 1).^2 .* (6 .- 4 .* eta))
    return E
end


"""
使用BM 拟合 E-V曲线
:param cl: cell
:param el: energy list
:param p0: 初始参数=[V0, B0, B0', E0]
"""
function BMfit(vl::Vector{Float64},el::Vector{Float64},cpcell::UnitCell,p0::Vector{Float64}=[1200.0, -10.0, 4.0, -1200.0])
    fit = curve_fit(birch_murnaghan, vl, el, p0)
    # 拟合结果
    fitted_params = fit.param
    min_lattice_constant = (fitted_params[1]/cpcell.copy[1]/cpcell.copy[2]/cpcell.copy[3])^(1/3)
    println("Fitted parameters:")
    println("V0 = ", fitted_params[1])
    println("B0 = ", fitted_params[2])
    println("B0' = ", fitted_params[3])
    println("E0 = ", fitted_params[4])
    println("latticeconstant = ",min_lattice_constant )
    fig = Figure()
    fitfunc(x)=birch_murnaghan(x, fitted_params)
    # 创建一个 Axis，并设置坐标范围
    ax = Axis(fig[1, 1],xlabel="Volume", ylabel="Energy", title="Birch-Murnaghan EOS Fit"
              )
    x=LinRange(minimum(vl),maximum(vl),1000)
    scatter!(ax,vl,el,color=:red,label="Data")
    lines!(ax,x,fitfunc.(x),color=:blue,label="Fit")
    axislegend(ax, position=:rt)
    fig
    return min_lattice_constant,fig
    end

end