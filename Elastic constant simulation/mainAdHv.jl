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

ct=3.0
interaction = Interaction(lj, Flj, ct, 0.1)

Ts=1.0
Ps=100.0
Tb=1.5*Ts
Pb=Ps*0.5
TQ=10
TW=1000
HooverChainN=4
dt=0.001
maxstep=1000000
cpc=[2,2,2]
dumpsequence=1
printsequence=10
projectname="AdHv_444_Ts=1_ps=100"




inicell=initcell(Pb,Tb,atoms,interaction,cp=cpc,Prg=[0.4,6.0])
println("inipressure=$(pressure_int(inicell,interaction)),temp=$(cell_temp(inicell))")
cell=deepcopy(inicell)
natom=length(inicell.atoms)

Qs=3*natom*Ts*(TQ*dt)^2
Ws=3*natom*Ts*(TW*dt)^2

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

