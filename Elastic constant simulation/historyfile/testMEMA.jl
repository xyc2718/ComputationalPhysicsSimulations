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
using Statistics
using Plots
using Test


"""
EMA.jl测试文件
"""

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


lattice_constant = 3.61
# 定义铜的FCC晶胞的基矢量
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

# 创建铜的原子列表
atoms = [Atom(pos) for pos in atom_positions]

cell=UnitCell(lattice_vectors,atoms)
cpcell=copycell(cell,3,3,3)
fcell=filtercell(cpcell)

println("test Cu cell:")

println()
println()
println("############################")
println("test for Frho:")
println([Frho(fcell,i) for i in 1:length(fcell.atoms)])
println("############################")
println("test for dFrho:")
println([dFrho(fcell,i) for i in 1:length(fcell.atoms)])
println("############################")
println("test for dFrho with Ngradiant:")
println([Ngradiant(fcell,i,Frho,[]) for i in 1:length(fcell.atoms)])