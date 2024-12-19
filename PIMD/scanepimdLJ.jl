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



projectname="scane PIMD For Lj fcc_dt=0.0001_fix"
ct=2.5
dt=0.0001 
maxstep=20000
dumpE=1
printsequence=100
dumpsequence=1
beginsamplestep=5000
beadrg=[64]
Tsrg=1.0:10.0:500.0
# Tsrg=[161.0]
cpc=[1,1,1]
inicell=filtercell(copycell(cell0,cpc...))
dr=0.000
t0=dt*0.1
function ULJ(r::Float64)
    return 4*(r^(-12)-r^(-6))
end
function FLJ(r::SVector{3,Float64})
    nr=norm(r)
    return 24*(2*nr^(-14)-nr^(-8))*r
end

interaction=Interaction(ULJ,FLJ,ct,0.01)
natom=length(inicell.atoms)


interactions=Interactions(interaction,inicell)
# minimizeEnergy!(inicell,interactions,rg=[1.2,6.0])
println("lattice_vectors=",inicell.lattice_vectors)
# fig=visualize_unitcell_atoms(inicell)
# display(fig)


basepath="outputScane\\$projectname"
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
    write(logfile,"IntergrateMethod:PIMD Langevin,Interaction:LJ")
    write(logfile, "$natom  atoms\n")
    write(logfile, "mass:$(inicell.atoms[1].mass/amuM)\n")
    write(logfile, "PIMD for Fcc LJ\n")
    write(logfile, "scanebeadrange=$beadrg\n")
    write(logfile, "scaneTrange=$beadrg\n")
    write(logfile, "Tsrg=$Tsrg\n")
    write(logfile, "cpsize=$cpc\n")
    write(logfile, "maxstep=$maxstep\n")
    write(logfile, "dt=$dt\n")
    write(logfile, "ct=$ct\n")
    write(logfile, "dr=$dr\n")
    write(logfile, "dumpsequence=$dumpsequence\n")
    write(logfile, "printsequence=$printsequence\n")
    write(logfile, "beginsamplestep=$beginsamplestep\n")
    write(logfile, "dumpE=$dumpE\n")
end


open("$basepath\\Log.json", "a") do file
    for N in beadrg
        for Ts in Tsrg
            bf=false
            
            TQ=10
            Qs=3*natom*Ts*kb*(TQ*dt)^2
            thermostat = Thermostat(Ts, Qs, 0.0, 0.0)

            if N==64 && Ts<405.0
                continue
            end


                mkpath("$basepath\\N=$N\\T=$Ts")
                open("$basepath\\N=$N\\T=$Ts\\Log.txt", "w") do io
                    # jldopen("$basepath\\DumpCell.JLD2","w") do iojl
                        cell=deepcopy(inicell)
                        z=cell2z(cell,thermostat)
                        # randcell!(cell,interactions,k=dr)
                        bdc=map2bead(cell,N,Ts,r=dr)
                        update_rmat!(cell)
                        update_fmat!(cell,interactions)
                        # pl,ql=get_bead_z(bdc)
                        # println(pl)
                        tempjs=""
                        for i in 1:maxstep
                            # pimdStep!(bdc,dt,Ts,interactions)
                            if N==1
                                RK3_step!(z,dt,cell,interactions,thermostat)
                            else
                                pimdLangevinStep!(bdc,dt,Ts,interactions,t0=t0)
                            # RK3_step!(z,dt,cell,interactions,thermostat)
                            end
                            if N==1
                                bdc=map2bead(cell,N,Ts,r=dr)
                            end
                            Ek=cell_Ek(bdc,interactions,Ts)
                            Ep=cell_energy(bdc,interactions)
                            Tt=cell_temp(bdc)
                            writedlm(io, [i, Ek, Ep,Ek+Ep,Tt]')
                            if Ek>1e2 || Ep>-200
                                println("!!!!!!!!!!\nWarning! Ek>1e2 with Ts=$Ts,N=$N,step=$i,this loop will break\n!!!!!!!!\n")
                                bf=true
                                break
                            end

                            if i>beginsamplestep
                                if mod(i,dumpE)==0

                                    data = Dict(
                                        "N" => N,
                                        "Ts"=> Ts,
                                        "Ek" => Ek,
                                        "Ep" =>Ep,
                                        "E"=>Ep+Ek,
                                        "Tt"=>Tt
                                    )
                                    tempjs*=JSON.json(data)
                                    tempjs*="\n"
                                    # write(file, JSON.json(data))
                                    # write(file, "\n")
                                    if mod(i,printsequence)==0
                                        # println("##############################################")
                                        println("step=$i")
                                        println(data)
                                        write(file,tempjs)
                                        tempjs=""
                                        println("##############################################")
                                    end
                                end
                            end     
                    
                        end
                    #end
                end
                GC.gc() 

                if bf 
                    println("\nexit loop of Ts=$Ts\n")
                    break
                end
        end

        GC.gc() 
    end
    # write(file,"\n]")
end