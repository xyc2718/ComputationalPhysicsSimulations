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
lattice_constant =4.6 #A

#HCP
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')
# HCP 晶胞原子位置 (正交晶格框架)
atom_positions = [
    Vector([0.0, 0.0, 0.0]),            # 原子 1
    Vector([0.5, 0.5, 0.0]),            # 原子 2
    Vector([0.0, 0.333, 0.5]),          # 原子 3
    Vector([0.5, 0.833, 0.5])           # 原子 4
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
r2=rOH+0.00
theta0=re["theta0"]
theta1=theta0+0.00
O=Atom(invlt*[0.0, 0.0, 0.0],15.9994*amuM)
H1=Atom(invlt*[0.0+r1 , 0.0, 0.0],1.008*amuM)
H2=Atom(invlt*[0.0+r2*cos(theta1), 0.0+r2*sin(theta1), 0.0],1.008*amuM)
atoms = [Atom(pos,100*amuM) for pos in atom_positions]
cell0=UnitCell(lattice_vectors, atoms)
structcell=filtercell(copycell(cell0,cpc...))

mol=Molecule([[1,2,3]],[O,H1,H2])
inicell,water=mapCell2Molecue(structcell::UnitCell,mol)




projectname="ICE_Classic_111_200K"
ensemble="NVE"
Ts=200.0 #K
Ps=0.0 #[p]
dt=0.0001 #ps
Tb=Ts
Pb=Ps
maxstep=10000
dumpsequence=1
printsequence=10
TQ=10
TW=500
cpc=[1,1,1]
ifcheckConvergence=true
interaction=TIP3P(water,cutCoulomb=3.0)

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
println("\nensemble:$ensemble\n")
##logfile
open("$basepath\\Config.txt", "w") do logfile
    write(logfile, "projectname=$projectname\n")
    write(logfile,"IntergrateMethod:RK3,Interaction:TIP3P")
    write(logfile, "ensemble:$ensemble\n")           

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
    write(logfile, "dumpsequence=$dumpsequence\n")
    write(logfile, "printsequence=$printsequence\n")
end

open("$basepath\\Log.txt", "w") do io
    jldopen("$basepath\\DumpCell.JLD2","w") do iojl
cell=deepcopy(inicell)

# for atom in cell.atoms 
#     atom.position+=0.1*randn(3)

# end
if ensemble=="NVE"
    z=cell2z(cell)
elseif ensemble=="NVT"
    z=cell2z(cell,thermostat)
elseif ensemble=="NPT"
    z=cell2z(cell,thermostat,barostat)
else
    throw(ArgumentError("ensemble not found"))
end
update_rmat!(cell)
update_fmat!(cell,interaction)

for i in 1:maxstep
if ensemble=="NVE"
    RK3_step!(z,dt,cell,interaction)
elseif ensemble=="NVT"
    RK3_step!(z,dt,cell,interaction,thermostat)
elseif ensemble=="NPT"
    RK3_step!(z,dt,cell,interaction,thermostat,barostat)
end
pint=pressure_int(cell,interaction)
T=cell_temp(cell)

# Ek=0.0
# for atom in cell.atoms
#     Ek+=0.5*norm(atom.momentum)^2/atom.mass
# end
# println("Ek=$Ek")
# Ep=cell_energy(cell,interaction)
# println("Ep=$Ep")
# println("E=$(Ep+Ek)")
# println(cell.atoms[8])
# println(cell.atoms[7])
if mod(i,dumpsequence)==0
writedlm(io, [i, T, pint,cell.Volume,barostat.Pv]')
write(iojl, "cell_$i", cell)
end
if mod(i,printsequence)==0
    println("step: $i, Temp: $T Pressure: $pint,Volume: $(cell.Volume), Rt :$(thermostat.Rt),Pt:$(thermostat.Pt),Pv:$(barostat.Pv)")
end

# if mod(i,1)==0
#     figi=visualize_unitcell_atoms(cell,sizelist=visize)
#     save("test1water1/$(lpad(i,3,'0')).png",figi)
# end

end
end
end