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


lattice_constant =4.7 #A

#HCP
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')
# HCP 晶胞原子位置 (正交晶格框架)
atom_positions = [
    Vector([0.0, 0.0, 0.0]),            # 原子 1
    Vector([0.5, 0.5, 0.0]),            # 原子 2
    Vector([0.0, 0.333, 0.5]),          # 原子 3
    Vector([0.5, 0.833, 0.5])           # 原子 4
]

cpc=[1,1,1]
para=getpara()
kb=para["kb"]
h=para["h"]
amuM=para["amuM"]
invlt=inv(lattice_vectors)
O=Atom(invlt*[0.5, 0.5, 0.5],16.0*amuM)
H1=Atom(invlt*[0.5+0.98343 , 0.0, 0.1],1.0*amuM)
H2=Atom(invlt*[0.5-0.15417, 0.1-0.62358, 0.0],1.0*amuM)
atoms = [Atom(pos,100*amuM) for pos in atom_positions]
cell0=UnitCell(lattice_vectors, atoms)
structcell=filtercell(copycell(cell0,cpc...))

mol=Molecule([[1,2,3]],[O,H1,H2])
wcell,water=mapCell2Molecue(structcell::UnitCell,mol)
apply_PBC!(wcell,water)
println(water.connection)
println(water.atoms)

interactions=TIP3P(water)

dt=0.0001
Ts=200.0
TQ=10.0
natom=length(wcell.atoms)
Qs=3*natom*Ts*kb*(TQ*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)
visize=ones(Float64,natom)
for k in eachindex(visize)
    if mod(k,3)==0
        visize[k]=3.0
    end
end

z=cell2z(wcell,thermostat)
# z=cell2z(wcell)
update_rmat!(wcell)
update_fmat!(wcell,interactions)
for step in 1:10000

    RK3_step!(z,dt,wcell,interactions,thermostat)
    # pint=pressure_int(wcell,interactions)
    T=cell_temp(wcell)
    println("step: $step, Temp: $T")
    if mod(step,1)==0
    figi=visualize_unitcell_atoms(wcell,sizelist=visize)
    save("test1water1/$(lpad(step,3,'0')).png",figi)
    end
end

readline()