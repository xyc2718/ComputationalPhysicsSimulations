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




kb=8.617332385e-5 #eV/K
amuM=1.03642701e-4 #[m]/amu
Mcu=63.546 #amu
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
atoms = [Atom(pos,Mcu*amuM) for pos in atom_positions]
#lj
function lj(r::Float64)
    return 4*(1/r^12-1/r^6)
end
function Flj(r::Vector{Float64})
    rn=norm(r)
    return 24*(2/rn^14-1/rn^8)*r
end




projectname="AdHvRk3_NVTrealUnit_444_Ts=1_p=100"
ct=3.0
Ts=300.0
Ps=0.0
dt=0.001
Tb=1.5*Ts
Pb=Ps
maxstep=50000
dumpsequence=1
printsequence=10
dumpcellsequence=100
TQ=10
TW=-1 #means no barostat
cpc=[2,2,2]
inicellmethod="Pb and Tb"

interaction = Interaction(lj, Flj, ct, 0.1)

if inicellmethod=="Pb and Tb"
    inicell=initcell(Pb,Tb,atoms,interaction,cp=cpc,Prg=[1.3,6.0])
else
    inicell=minEenergyCell(Tb,atoms,interaction,cpc)
    inicellmethod="min energy with Tb"
end


println("initemp=$(cell_temp(inicell))")
println("inipressure=$(pressure_int(inicell,interaction))")
println(inicell.lattice_vectors)
natom=length(inicell.atoms)
Qs=3*natom*kb*Ts*(TQ*dt)^2
Ws=3*natom*kb*Ts*(TW*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)
barostat = Barostat(Ps, Ws, 0.0, 0.0)



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
end

open("$basepath\\Log.txt", "w") do io
    jldopen("$basepath\\DumpCell.JLD2","w") do iojl
cell=deepcopy(inicell)
z=cell2z(cell,thermostat);
for i in 1:maxstep
RK3_step!(z,dt,cell,interaction,thermostat)
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