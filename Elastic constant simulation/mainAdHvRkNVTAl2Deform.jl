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
using JSON
using base.Threads
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



projectname="AdHvRk3_NVT_Al2_Deform_T=300K_P=0Gpa"

ct=6.5
Ts=300.0
V0=532.0867533091118
Ps=0.0
dt=0.001
Tb=1.5*Ts
Pb=Ps
maxstep=10000
dumpsequence=1
printsequence=100
dumpcellsequence=1000
dumpForceTensorSequence=1
TQ=10
TW=-1 #means no barostat
cpc=[1,1,1]
flaglist=[0,1,2,3,4,5,6]
deltalist=[-0.01,0.01]


inicellmethod="Pb and Tb"
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


lt0=inicell.lattice_vectors*(V0/inicell.Volume)^(1/3)
set_lattice_vector!(inicell,lt0,interaction)


println("initemp=$(cell_temp(inicell))")
println("inipressure=$(pressure_int(inicell,interaction))")
println("inicell Volume=$(inicell.Volume)")
println("inicell Vector=",inicell.lattice_vectors)
natom=length(inicell.atoms)
Qs=3*natom*kb*Ts*(TQ*dt)^2
Ws=3*natom*kb*Ts*(TW*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)
barostat = Barostat(Ps, Ws, 0.0, 0.0)



    basepath="outputNVT\\$projectname"
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

open("$basepath\\Log.json", "w") do file
    for flag in flaglist
        for delta in deltalist
            mkpath("$basepath\\flag=$flag\\delta=$delta")
            open("$basepath\\flag=$flag\\delta=$delta\\Log.txt", "w") do io
                jldopen("$basepath\\flag=$flag\\delta=$delta\\DumpCell.JLD2","w") do iojl
                    cell=deepcopy(inicell)
                    dmat=deform_mat(flag,delta)
                    deform_cell!(cell,dmat,interaction)
                    z=cell2z(cell,thermostat);
                    for i in 1:maxstep
                        RK3_step!(z,dt,cell,interaction,thermostat)
                        if mod(i,dumpsequence)==0
                            pint=pressure_int(cell,interaction)
                            T=cell_temp(cell)
                            writedlm(io, [i, T, pint,cell.Volume,barostat.Pv]')
                        end
                        if mod(i,printsequence)==0
                            pint=pressure_int(cell,interaction)
                            T=cell_temp(cell)
                            println("flag=$flag,delta=$delta,$step: $i, Temp: $T Pressure: $pint,Volume: $(cell.Volume), Rt :$(thermostat.Rt),Pt:$(thermostat.Pt),Pv:$(barostat.Pv)")
                        end
                        if mod(i,dumpForceTensorSequence)==0
                            ft=force_tensor(cell,interaction)
                            dUdh=dUdhij(cell,interaction,BigFloat("1e-8"))
                            ft0=ft-(dUdh*transpose(cell.lattice_vectors))./cell.Volume
                            T=cell_temp(cell)
                            data = Dict(
                                "flag" => flag,
                                "delta"=> delta,
                                "forcetensor0" => ft,
                                "dUdhij" => dUdh,
                                "hij"=>cell.lattice_vectors,
                                "forcetensor"=>ft0,
                                "Temp"=>T
                            )
                            write(file, JSON.json(data))
                            if mod(i,printsequence)==0
                                println("##############################################")
                                println(data)
                                println("##############################################")
                            end
                        end     
                    end
                end
            end
        end
    end
end
