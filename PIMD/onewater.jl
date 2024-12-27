using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using LsqFit
include("src\\Elastic.jl")
using .Elastic
using FFMPEG
using DelimitedFiles
using Distributions
using JLD2
using Plots

using Base.Threads
using Random
Random.seed!(255515)
println("Number of threads: ", Threads.nthreads())
kb=8.617332385e-5 #eV/K
amuM=1.03642701e-4 #[m]/amu
MAl=26.9815385 #amu
P00=160.2176565 #Gpa/[p]
lattice_constant =25.0 #A

#HCP
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')
# HCP 晶胞原子位置 (正交晶格框架)
atom_positions = [
    Vector([0.0, 0.0, 0.0]),            # 原子 1
]


cpc=[1,1,1]
para=getpara()
kb=para["kb"]
h=para["h"]
amuM=para["amuM"]
invlt=inv(lattice_vectors)
re=getparatip3p()
rOH=re["rOH"]
r1=rOH
r2=rOH+0.01
theta0=re["theta0"]
theta1=theta0+0.01
O=Atom(invlt*[0.0, 0.0, 0.0],15.9994*amuM)
H1=Atom(invlt*[0.0+r1 , 0.0, 0.0],1.008*amuM)
H2=Atom(invlt*[0.0+r2*cos(theta1), 0.0+r2*sin(theta1), 0.0],1.008*amuM)
atoms = [Atom(pos,100*amuM) for pos in atom_positions]
structcell=UnitCell(lattice_vectors, atoms)

mol=Molecule([[1,2,3]],[O,H1,H2])
inicell,water=mapCell2Molecue(structcell::UnitCell,mol)


projectname="H2O_111"
calculate_gr=true
grsequence=1
beginsamplegr=2000
gr=RadialDistribution(0.7,1.2,200)

Ts=991.0 #K
Ps=0.0 #[p]
dt=0.00025 #ps
t0=0.5
N=1
Tb=Ts
Pb=Ps
maxstep=10000
dumpsequence=1
dumpcellsequence=1000
beginsamplestep=1
printsequence=100
TQ=5
TW=1000
cutCoulomb=4.0
cpc=[1,1,1]
ifcheckConvergence=true
interaction=TIP3P(water,cutCoulomb=cutCoulomb)

println("initemp=$(cell_temp(inicell))")
println("inipressure=$(pressure_int(inicell,interaction))")
println(inicell.lattice_vectors)
natom=length(inicell.atoms)
Qs=3*natom*Ts*kb*(TQ*dt)^2
Ws=3*natom*Ts*kb*(TW*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)
barostat=Barostat(Ps,Ws,inicell.Volume,0.0)
visize=ones(Float64,natom)

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
    write(logfile,"IntergrateMethod:PIMD,Interaction:TIP3P")
    write(logfile, "N:$N\n")    
    write(logfile, "t0:$t0\n")        
    write(logfile, "$natom  atoms\n")
    write(logfile, "cutCoulomb=$cutCoulomb\n")
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
    write(logfile, "dumpsequence=$dumpsequence\n")
    write(logfile, "dumpcellsequence=$dumpsequence\n")
    write(logfile, "printsequence=$printsequence\n")
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
        # LangevinVerlet_step!(dt,cell,interaction,Ts,t0)
        RK3_step!(z,dt,cell,interaction)
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
    if i>beginsamplestep
    if calculate_gr
        if mod(i,grsequence)==0
            calculate_gr!(gr,cell,1,2)
        end
    end
    end

end
end
end

if calculate_gr
    Normalize_gr!(gr)
    p=Plots.plot(gr.rm,gr.ngr,label="g(r)",xlabel="r",ylabel="g(r)",title="g(r) of N=$N,T=$Ts",lw=2,dpi=600)
    Plots.savefig(p,"$basepath\\gr.png")
    jldopen("$basepath\\DumpCellgr.JLD2","w") do jlgr
        write(jlgr,"gr",gr)
    end

    
end