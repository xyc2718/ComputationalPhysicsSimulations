using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using LsqFit
include("Elastic.jl")
using .Elastic
using FFMPEG
using DelimitedFiles
using Distributions
using JLD2
# 定义铜的晶格常数（单位：Å）
lattice_constant = 1.0

# 定义铜的FCC晶胞的基矢量
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

# 定义铜的FCC晶胞中的原子位置（单位：Å）
atom_positions = [
    Vector([0.0, 0.0, 0.0]),
    Vector([0.0, 0.5, 0.5]),
    Vector([0.5, 0.0, 0.5]),
    Vector([0.5, 0.5, 0.0]),
    Vector([1.0, 0.0, 0.0]),
    Vector([0.0, 1.0, 0.0]),
    Vector([0.0, 0.0, 1.0]),
    Vector([0.5, 1.0, 0.5]),
    Vector([1.0, 0.5, 0.5]),
    Vector([0.5, 0.5, 1.0]),
    Vector([1.0, 0.0, 1.0]),
    Vector([1.0, 1.0, 0.0]),
    Vector([0.0, 1.0, 1.0]),
    Vector([1.0, 1.0, 1.0])
] 

# 创建铜的原子列表
atoms = [Atom(pos) for pos in atom_positions]

#lj势能
function lj(r::Float64)
    return 4*(1/r^12-1/r^6)
end
function Flj(r::Vector{Float64})
    rn=norm(r)
    return 24*(2/rn^14-1/rn^8)*r
end

ct=5.0
interaction = Interaction(lj, Flj, ct, 0.1)

x=1:0.001:ct
y=interaction.cutenergy.(x)
lines(x,y)



Ts=1.0
Ps=100.0

fcell=initcell(Ps,Ts,atoms,interaction,cp=[3,3,3],Prg=[0.03,8])
println(pressure_int(fcell,interaction))
println(cell_temp(fcell))
println(fcell.lattice_vectors)

barostat=Barostat(Ps,0.01,fcell.Volume,0.0)
open("dump1.txt", "w") do io
    jldopen("dumpcell.jld2", "w") do file
for i in 1:1000000
    dt=0.001
    n=4
    gamma0=0.1
    gammav=0.001
    LA_step!(fcell,interaction,dt,Ts,barostat,gamma0,gammav,n)
    if i%100==0
        z=[i,(pressure_int(fcell,interaction)),(fcell.Volume),(cell_temp(fcell)),(barostat.Pv)]
        writedlm(io, z')
        println("step $i,P=$(z[2]),V=$(z[3]),T=$(z[4]),Rv=$(z[5])")
        write(file, "cell_$i", fcell)
    end
end
end
end        