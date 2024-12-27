using StaticArrays
using Plots
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
rdseed=25515
Random.seed!(25515)
println("Number of threads: ", Threads.nthreads())

n0 = 100
range_min = -0.8
range_max = -0.7


lattice_constant = 25.0 #A
# atom_positions = [
# Vector([-0.75,0.0,0.0])./lattice_constant
# ]
n=100
atom_positions = [Vector([rand(-1.0:0.01:1.0), rand(-1.0:0.01:1.0), 0.0]) ./ lattice_constant for _ in 1:n]
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

para=getpara()
kb=para["kb"]
h=para["h"]
amuM=para["amuM"]

atoms = [Atom(pos,1*amuM) for pos in atom_positions]

inicell=UnitCell(lattice_vectors, atoms)



projectname="DoubleWellF"
ifcalculate_pr=true

trajid=1
trajsequence=100
prsequence=1
pr=SpatialDistribution(inicell,200,200,1,rmin=([-1.0,-1.0,-lattice_constant]/lattice_constant),rmax=([1.0,1.0,lattice_constant]/lattice_constant))
ct=2.5
Ts=200.0
dt=0.00025
beginsamplestep=1000
maxstep=100000
dumpcellsequence=100000
dumpsequence=100
printsequence=10
N=16

# Traj=Trajectory(beginsamplestep,maxstep,trajsequence,dt)
w=50.0
# w=3800/1e8
A=-0.46323262878163146
B=0.41504545929904935

println("PIMD beads: $N")
cpc=[1,1,1]

dr=0.000
t0=0.1
function U(r::SVector{3,Float64})
    x,y,z=r
    para=getpara()
    amuM=para["amuM"]
    return A*x^2+B*x^4+0.5*amuM*w^2*y^2+0.5*amuM*w^2*z^2
end

function F(r::SVector{3,Float64})
    x,y,z=r
    para=getpara()
    amuM=para["amuM"]
    return [-2*A*x-4*B*x^3,-amuM*w^2*y,-amuM*w^2*z]
end

field=Field(U,F)
nb=Vector{Neighbor}([Neighbor()])
inter=Vector{AbstractInteraction}([field])
interaction=Interactions(inter,nb)
natom=length(inicell.atoms)
TQ=1
Qs=3*natom*Ts*kb*(TQ*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)

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

##logfile
open("$basepath\\Config.txt", "w") do logfile
    write(logfile, "projectname=$projectname\n")
    write(logfile,"IntergrateMethod:PIMD/Langevin,Interaction:DoubleWell")
    write(logfile,"RandomSeed=$rdseed\n")
    write(logfile, "N:$N\n")    
    write(logfile, "t0:$t0\n")        
    write(logfile, "$natom  atoms\n")
    write(logfile, "m=$(inicell.atoms[1].mass)\n")
    write(logfile, "Ts=$Ts\n")
    write(logfile, "Qs=$Qs\n")
    write(logfile, "TQ=$TQ\n")
    write(logfile, "cpsize=$cpc\n")
    write(logfile, "maxstep=$maxstep\n")
    write(logfile, "dt=$dt\n")
    write(logfile, "dumpsequence=$dumpsequence\n")
    write(logfile, "dumpcellsequence=$dumpsequence\n")
    write(logfile, "printsequence=$printsequence\n")
    write(logfile, "beginsamplestep=$beginsamplestep\n")
    write(logfile,"\n\nDoubleWell Parament:\n\n")
    write(logfile,"w=$w\n")
    write(logfile,"A=$A\n")
    write(logfile,"B=$B\n")
end
println(interaction.neighbors)

open("$basepath\\Log.txt", "w") do io
    jldopen("$basepath\\DumpCell.JLD2","w") do iojl

if N==1
    cell=deepcopy(inicell)
    z=cell2z(cell)
    update_rmat!(cell)
    update_fmat!(cell,interaction)
elseif (N>=4)&&(mod(N,2)==0)
    cell=map2bead(inicell,interaction,N,Ts,r=0.0)
else
    throw("Wrong Beads Number")

end




for i in 1:maxstep
    # println(cell.fmat)
    # println(cell.atoms[2].momentum/cell.atoms[2].mass)
    if N==1
        # RK3_step!(z,dt,cell,interaction,thermostat)
        LangevinVerlet_step!(dt,cell,interaction,Ts,t0)
    else
        pimdLangevinStep!(cell,dt,Ts,interaction,t0=t0)
    end
    provide_cell(cell,dt)
    T=cell_temp(cell)
    Ek=cell_Ek(cell,interaction,Ts)
    Ep=cell_energy(cell,interaction)
    
    if mod(i,dumpsequence)==0
    writedlm(io, [i,Ek,Ep,Ek+Ep,T]')
    end

    if mod(i,printsequence)==0
        println("step: $i, Ek: $Ek Ep: $Ep,E:$(Ek+Ep),T:$T,Rt:$(thermostat.Pt)")
    end
    if mod(i,dumpcellsequence)==0
        write(iojl, "cell_$i", cell)
    end
    # if mod(i,trajsequence)==0
    #     calculate_traj!(Traj,cell,trajid)
    # end
    if i>beginsamplestep
        # ri=get_position0(cell,1)
        # println(get_position0(cell,1))
        # println(ri)
        # println(pr.rmin)
        # println(pr.rmax)
        # println(all(ri.<pr.rmax))
        # println(all(ri.>pr.rmin))
        if ifcalculate_pr  
            if mod(i,prsequence)==0
                calculate_pr!(pr,cell)
            end
        end
    end


end
end
end

if ifcalculate_pr
    Normalize_pr!(pr)
    fig=Plots.heatmap(pr.xl.*lattice_constant,pr.yl.*lattice_constant,transpose(pr.npr[:,:,1]),color=:viridis,title="Spatial Distribution Beads=$N,T=$Ts K",xlabel="x/A",ylabel="y/A")
    savefig(fig,"$basepath\\SpatialDistribution.png")
    jldopen("$basepath\\SpatialDistribution.JLD2","w") do iojl
        write(iojl,"pr",pr)
    end
end

# fig=Plots.plot(Traj.rl[1,:],Traj.rl[2,:],xlabel="x/A",ylabel="y/A",title="Trajectory of Atom $trajid Beads=$N,T=$Ts")
# savefig(fig,"$basepath\\Trajectory.png")
# jldopen("$basepath\\Trajectory.JLD2","w") do iojl
#     write(iojl,"Traj",Traj)
# end