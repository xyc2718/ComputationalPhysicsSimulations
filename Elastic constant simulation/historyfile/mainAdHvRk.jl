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
#lj势能
function lj(r::Float64)
    return 4*(1/r^12-1/r^6)
end
function Flj(r::SVector{3}{Float64})
    rn=norm(r)
    return 24*(2/rn^14-1/rn^8)*r
end




projectname="AdHvRk3_222metalunit_Ts=1_p=100"
ct=6.5
Ts=300.0 #K
Ps=0.0 #[p]
dt=0.001 #ps
Tb=Ts
Pb=Ps
maxstep=100000
dumpsequence=1
printsequence=10
TQ=10
TW=1000
cpc=[1,1,1]
inicellmethod="Pb and Tb"

interaction = Interaction(lj, Flj, ct, 0.1)

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
Qs=3*natom*Ts*kb*(TQ*dt)^2
Ws=3*natom*Ts*kb*(TW*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)
barostat=Barostat(Ps,Ws,inicell.Volume,0.0)


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
    write(logfile, "inicellmethod=$inicellmethod\n")
end

open("$basepath\\Log.txt", "w") do io
    jldopen("$basepath\\DumpCell.JLD2","w") do iojl
cell=deepcopy(inicell)
z=cell2z(cell,thermostat,barostat);
for i in 1:maxstep
RK3_step!(z,dt,cell,interaction,thermostat,barostat)
pint=pressure_int(cell,interaction)
T=cell_temp(cell)
if mod(i,dumpsequence)==0
writedlm(io, [i, T, pint,cell.Volume,barostat.Pv]')
write(iojl, "cell_$i", cell)
end
if mod(i,printsequence)==0
    println("step: $i, Temp: $T Pressure: $pint,Volume: $(cell.Volume), Rt :$(thermostat.Rt),Pt:$(thermostat.Pt),Pv:$(barostat.Pv)")
end

end
end
end