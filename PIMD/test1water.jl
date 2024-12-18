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


lattice_constant =1.0 #A

# 定义铜的FCC晶胞的基矢量
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')
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

cpc=[1,1,1]
para=getpara()
kb=para["kb"]
h=para["h"]
amuM=para["amuM"]
O=Atom([0.0, 0.0, 0.0],16.0*amuM)
H1=Atom([0.0, 0.0, 0.1],1.0*amuM)
H2=Atom([0.0, 0.1, 0.0],1.0*amuM)
atoms = [Atom(pos,100*amuM) for pos in atom_positions]
cell0=UnitCell(lattice_vectors, atoms)
structcell=filtercell(copycell(cell0,cpc...))

mol=Molecule([[1,2,3]],[O,H1,H2])
wcell,water=mapCell2Molecue(structcell::UnitCell,mol)
apply_PBC!(wcell,water)
println(water.connection)
println(water.atoms)
# fig=visualize_unitcell_atoms(wcell)
# display(fig)
# readline()
conOH=Vector{Vector{Int}}([])
for cn in water.connection
    for i in 2:length(cn)
    push!(conOH,[cn[1],cn[i]])
    end
end

function getparatip3p()
    kk=0.0433634
    return Dict(
        "kOH" => kk*450,    # eV/A
        "kHOH" => 55.0*kk,   # [m]/amu
        "rOH" => 0.9572,       # amu
        "theta0" => 104.52*pi/180,      # GPa/[p]
        "h" => 6.582119281e-4      # eV*ps
    )
    
end

function EOH(r::SVector{3,Float64})
    pare=getparatip3p()
    k=pare["kOH"]
    r0=pare["rOH"]
    return k*(norm(r)-r0)^2
end
function FOH(r::SVector{3,Float64})
    nr=norm(r)
    pare=getparatip3p()
    k=pare["kOH"]
    r0=pare["rOH"]
    return (2*k*(nr-r0)^2*(r/nr),-2*k*(nr-r0)^2*(r/nr))
end

bondOH=Bond(conOH,EOH,FOH)

conHOH=water.connection

function EHOH(r1::SVector{3,Float64},r2::SVector{3,Float64})
    
    pare=getparatip3p()
    k=pare["kHOH"]
    theta0=pare["theta0"]
    nr1=norm(r1)
    nr2=norm(r2)
    cs=LinearAlgebra.dot(r1,r2)/nr1/nr2
    if cs>1.0
        cs=1.0
    end
    if cs<-1.0
        cs=-1.0
    end
    theta=acos(cs)
    return k*(theta-theta0)^2
end
function FHOH(r1::SVector{3,Float64},r2::SVector{3,Float64})
    theta0=107*pi/180
    pare=getparatip3p()
    k=pare["kHOH"]
    theta0=pare["theta0"]
    nr1=norm(r1)
    nr2=norm(r2)
    cs=LinearAlgebra.dot(r1,r2)/nr1/nr2
    n=LinearAlgebra.cross(r1,r2)
    t1=LinearAlgebra.cross(r1,n)
    t2=LinearAlgebra.cross(n,r2)
    t1=t1/norm(t1)
    t2=t2/norm(t2)
    if cs>1.0
        cs=1.0
    end
    if cs<-1.0
        cs=-1.0
    end
    theta=acos(cs)
    return (-2*k*(theta-theta0)*t1,-2*k*(theta-theta0)*t2)
end

angleHOH=Angle(conHOH,EHOH,FHOH)
nb=Vector{Neighbor}([Neighbor(),Neighbor()])
interactionlist=Vector{AbstractInteraction}([bondOH,angleHOH])
interactions=Interactions(interactionlist,nb)

dt=0.0005
z=cell2z(wcell)
update_rmat!(wcell)
update_fmat!(wcell,interactions)
for step in 1:10000

    RK3_step!(z,dt,wcell,interactions)
    # pint=pressure_int(wcell,interactions)
    T=cell_temp(wcell)
    println("step: $step, Temp: $T")
end
fig=visualize_unitcell_atoms0(wcell,iftext=true)
display(fig)
readline()