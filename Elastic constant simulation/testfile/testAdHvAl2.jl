using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using LsqFit
include("..\\src\\Elastic.jl")
using .Elastic
using FFMPEG
using DelimitedFiles
using Distributions
using JLD2
using Base.Threads

println("Number of threads: ", Threads.nthreads())

kb=8.617332385e-5 #eV/K
amuM=1.03642701e-4 #[m]/amu
MAl=26.9815385 #amu
P00=160.2176565 #Gpa/[p]
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
atoms = [Atom(pos,MAl*amuM) for pos in atom_positions]

lattice_constant = 4.032 #A

# 定义铜的FCC晶胞的基矢量
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')


ct=6.5
embeddingAl2=Embedding(embedding_energyAl2, embedding_forceAl2i)
interaction=Interaction(EMAAl2_phi, EMAAl2_phi_gradient, ct, 0.1, embeddingAl2)
Ts=500.0
Ps=0.0
Tb=Ts*10
Pb=Ps
TQ=10
TW=1000
HooverChainN=3
dt=0.001
maxstep=10000
cpc=[1,1,1]
dumpsequence=1
printsequence=10
projectname="AdHv_Al2_111_500K_test"
comment=""

inicellmethod="min energy"

embeddingAl2=Embedding(embedding_energyAl2, embedding_forceAl2i)
interaction=Interaction(EMAAl2_phi, EMAAl2_phi_gradient, ct, 0.1, embeddingAl2)


if inicellmethod=="Pb and Tb"
    inicell=initcell(Pb,Tb,atoms,interaction,cp=cpc,Prg=[1.3,6.0])
elseif inicellmethod=="min energy with Tb"
    inicell=minEenergyCell(Tb,atoms,interaction,cpc)
    inicellmethod="min energy with Tb"
elseif inicellmethod=="min energy"
    cell=UnitCell(lattice_vectors,atoms)
    cpcell=copycell(cell,cpc...)
    inicell=filtercell(cpcell)
    minimizeEnergy!(inicell,interaction,rg=[3.8,4.2])
else
    throw("method error")
end

println("inipressure=$(pressure_int(inicell,interaction)),temp=$(cell_temp(inicell))")
cell=deepcopy(inicell)
natom=length(inicell.atoms)

Qs=3*natom*Ts*kb*(TQ*dt)^2
Ws=3*natom*Ts*kb*(TW*dt)^2

thermostatchain = [Thermostat(Ts, Qs, 0.0, 0.0) for i in 1:HooverChainN]
for i in eachindex(thermostatchain)
    if i > 1 
        thermostatchain[i].Q = Qs / natom / 3
    end
end
# println(thermostatchain)
barostat=Barostat(Ps,Ws,cell.Volume,0.0)

# if !isdir("output\\$projectname")
#     mkpath("output\\$projectname")
#     println("Directory $projectname created.\n")
# else
#     println("Directory $projectname already exists.\n")
# end


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



##logfile
open("$basepath\\Config.txt", "w") do logfile
    write(logfile, "projectname=$projectname\n")
    write(logfile,"IntergrateMethod:Liouville,Interaction:LJ\n")
    write(logfile, "$natom  atoms\n")
    write(logfile, "Ts=$Ts\n")
    write(logfile, "Ps=$Ps\n")
    write(logfile, "Tb=$Tb\n")
    write(logfile, "Pb=$Pb\n")
    write(logfile, "Qs=$Qs\n")
    write(logfile, "Ws=$Ws\n")
    write(logfile, "Number of HooverChain=$HooverChainN\n")
    write(logfile, "TQ=$TQ\n")
    write(logfile, "TW=$TW\n")
    write(logfile, "cpsize=$cpc\n")
    write(logfile, "maxstep=$maxstep\n")
    write(logfile, "dt=$dt\n")
    write(logfile, "ct=$ct\n")
    write(logfile, "dumpsequence=$dumpsequence\n")
    write(logfile, "printsequence=$printsequence\n")
end


open("$basepath\\Log.txt", "w") do io
    jldopen("$basepath\\DumpCell.JLD2","w") do iojl
update_rmat!(cell)
update_fmat!(cell,interaction)
for step in 1:maxstep

    Andersen_Hoover_NPT_step!(cell,interaction,thermostatchain,barostat,dt,nresn=3)
    pint=pressure_int(cell,interaction)
    T=cell_temp(cell)
    if mod(step,dumpsequence)==0
    writedlm(io, [step, T, pint, cell.Volume, barostat.Pv]')
    write(iojl, "cell_$step", cell)
    end
    if mod(step,printsequence)==0
        println("step: $step, Temp: $T Pressure: $pint,Volume: $(cell.Volume),Pv:$(barostat.Pv)")
    end


end
end
end

