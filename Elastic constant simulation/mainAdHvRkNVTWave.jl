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

atoms = [Atom(pos,MAl*amuM) for pos in atom_positions]


lattice_constant = 4.032 #A
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

projectname="AdHvRk3_NVE_Al2_Wave"
ct=6.5
Ts=300.0
Ps=0.0
dt=0.001
Tb=1.0*Ts
Pb=Ps
# V0=2174.0130623648565
maxstep=2000
dumpsequence=1
printsequence=10
dumpcellsequence=1
TQ=10
TW=-1 #means no barostat
cpc=[1,4,1]
inicellmethod="min energy"
waveVector=[0.0,0.00,0.01]

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

 


println("initemp=$(cell_temp(inicell))")
println("inipressure=$(pressure_int(inicell,interaction))")
println(inicell.lattice_vectors)
natom=length(inicell.atoms)
Qs=3*natom*kb*Ts*(TQ*dt)^2
Ws=3*natom*kb*Ts*(TW*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)
barostat = Barostat(Ps, Ws, 0.0, 0.0)

# lt0=inicell.lattice_vectors.*((V0/inicell.Volume)^(1/3))


# set_lattice_vector!(inicell,lt0,interaction)

# println("V=$(inicell.Volume)")
# println("V0=$V0")
# println(inicell.cp)
# fig=visualize_unitcell_atoms0(inicell,iftext=true)
# display(fig)
# readline() 
applyWavePerturbation!(inicell,interaction,waveVector)


basepath="outputWave\\$projectname"
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
    write(logfile,"IntergrateMethod:RK3,Interaction:LJ")
    write(logfile, "$natom  atoms\n")
    write(logfile, "Md for Fcc Cu\n")
    write(logfile, "Ts=$Ts\n")
    write(logfile, "Ps=$Ps\n")
    write(logfile, "Tb=$Tb\n")
    write(logfile, "Pb=$Pb\n")
    write(logfile, "Qs=$Qs\n")
    write(logfile, "Ws=$Ws\n")
    write(logfile, "TQ=$TQ\n")
    write(logfile, "TW=$TW\n")
    write(logfile, "cpsize=$cpc\n")
    write(logfile, "maxstep=$maxstep\n")
    write(logfile, "dt=$dt\n")
    write(logfile, "ct=$ct\n")
    write(logfile, "dumpsequence=$dumpsequence\n")
    write(logfile, "printsequence=$printsequence\n")
    write(logfile, "dumpcellsequence=$dumpcellsequence\n")
    write(logfile, "WaveVector=$waveVector\n")
end

open("$basepath\\Log.txt", "w") do io
    jldopen("$basepath\\DumpCell.JLD2","w") do iojl
cell=deepcopy(inicell)
z=cell2z(cell);
update_rmat!(cell)
for i in 1:maxstep
RK3_step!(z,dt,cell,interaction)
pint=pressure_int(cell,interaction)
T=cell_temp(cell)
if mod(i,dumpsequence)==0
writedlm(io, [i, T, pint,cell.Volume,barostat.Pv]')
end
if mod(i,printsequence)==0
    println("step: $i, Temp: $T Pressure: $pint,Volume: $(cell.Volume), Rt :$(thermostat.Rt),Pt:$(thermostat.Pt),Pv:$(barostat.Pv)")
end
if mod(i,dumpcellsequence)==0
write(iojl, "cell_$i", cell)
end

end
end
end