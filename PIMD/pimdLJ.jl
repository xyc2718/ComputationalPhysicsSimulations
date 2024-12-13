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
using Random
Random.seed!(255515)
println("Number of threads: ", Threads.nthreads())

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


lattice_constant = 1.6 #A

lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

para=getpara()
kb=para["kb"]
h=para["h"]
amuM=para["amuM"]

atoms = [Atom(pos,100*amuM) for pos in atom_positions]

cell0=UnitCell(lattice_vectors, atoms)



projectname="PIMD For Lj fcc"
ct=2.5
Ts=150.0
dt=0.001 
maxstep=1000
dumpsequence=1
printsequence=100
N=4

#dt=0.0001,t0=100dt

println("PIMD beads: $N")
cpc=[1,1,1]
inicell=filtercell(copycell(cell0,cpc...))
dr=0.000
t0=dt
function ULJ(r::Float64)
    return 4*(r^(-12)-r^(-6))
end
function FLJ(r::SVector{3,Float64})
    nr=norm(r)
    return 24*(2*nr^(-14)-nr^(-8))*r
end

interaction=Interaction(ULJ,FLJ,ct,0.01)
natom=length(inicell.atoms)

TQ=1
Qs=3*natom*Ts*kb*(TQ*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)

interactions=Interactions(interaction,inicell)
# minimizeEnergy!(inicell,interactions,rg=[1.2,6.0])
println("lattice_vectors=",inicell.lattice_vectors)
# fig=visualize_unitcell_atoms(inicell)
# display(fig)



basepath="output\\$projectname"
if !isdir(basepath)
    mkpath(basepath)
    println("Directory $basepath created.\n")
else
    local counter = 1
    local newpath = basepath * "_$counter"
    while isdir(newpath)
        counter += 1
        newpath = basepath * "_$counter"
    end
    mkpath(newpath)
    println("Directory exists,new Directories $newpath created.\n")
    basepath=newpath
end


open("$basepath\\Config.txt", "w") do logfile
    write(logfile, "projectname=$projectname\n")
    write(logfile,"IntergrateMethod:PIMD,Interaction:LJ")
    write(logfile, "$natom  atoms\n")
    write(logfile, "PIMD for Fcc LJ\n")
    write(logfile, "Nbead=$N\n")
    write(logfile, "Ts=$Ts\n")
    write(logfile, "cpsize=$cpc\n")
    write(logfile, "maxstep=$maxstep\n")
    write(logfile, "dt=$dt\n")
    write(logfile, "ct=$ct\n")
    write(logfile, "dr=$dr\n")
    write(logfile, "dumpsequence=$dumpsequence\n")
    write(logfile, "printsequence=$printsequence\n")
end




open("$basepath\\Log.txt", "w") do io
    jldopen("$basepath\\DumpCell.JLD2","w") do iojl
cell=deepcopy(inicell)
    z=cell2z(cell,thermostat);
    randcell!(cell,interactions,k=dr)
    bdc=map2bead(cell,N,Ts,r=dr)
    update_rmat!(cell)
    update_fmat!(cell,interactions)
    # pl,ql=get_bead_z(bdc)
    # println(pl)
    println(bdc.cells[1].atoms[10].mass)
    println(bdc.cells[end].atoms[10].mass)
    betan=1/N/kb/Ts
    wn=1/betan/h
    println("wn=$wn")
    println("betan=$betan")

    for i in 1:maxstep
    # pimdStep!(bdc,dt,Ts,interactions)
    pimdLangevinStep!(bdc,dt,Ts,interactions,t0=t0)
    # RK3_step!(z,dt,cell,interactions,thermostat)
    
    if (mod(i,dumpsequence)==0)||(mod(i,printsequence)==0)
        # T=cell_temp(bdc)
        Ek=cell_Ek(bdc,interactions,Ts)
        Ep=cell_energy(bdc,interactions)
        Tt=cell_temp(bdc)
        # T=cell_temp(cell)
        # E=cell_energy(cell,interactions)
    end
    if mod(i,dumpsequence)==0
    writedlm(io, [i, Ek, Ep,Ek+Ep,Tt]')
    # write(iojl, "cell_$i", cell)
    end
    if mod(i,printsequence)==0
        println("step: $i, Ek: $Ek,", "Ep: $Ep","E:",Ek+Ep,"T:",Tt)
        # pl,ql=get_bead_z(bdc)
        # println(pl)
    end
    
    
    
end
fig=visualize_beadcell(bdc)
    display(fig)
    readline()
end
end