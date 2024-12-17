using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using LsqFit
include("src\\Elastic.jl")
using .Elastic
using FFMPEG
using DelimitedFiles
using Distributions
using JLD2
using Base.Threads
using Plots
using Random

O=Atom([0.0, 0.0, 0.0],6.0)
H1=Atom([0.0, 0.0, 1.0],1.0)
H2=Atom([0.0, 1.0, 0.0],1.0)
lattice_constant =1.0 #A

# 定义铜的FCC晶胞的基矢量
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')