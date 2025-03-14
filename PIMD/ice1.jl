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

println("Number of threads: ", Threads.nthreads())
kb=8.617332385e-5 #eV/K
amuM=1.03642701e-4 #[m]/amu
MAl=26.9815385 #amu
P00=160.2176565 #Gpa/[p]
lattice_constant =6.1229 #A

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

filename = "h2o-32.xyz"  # 替换为实际文件名
lines = readlines(filename)

# 初始化存储
atom_data = Vector{Vector{Float64}}([])  # 存储原子类型和坐标的数组

# 解析每一行
for line in lines
    # 跳过注释行和空行
    if startswith(line, "#") || isempty(strip(line))
        continue
    end
    
    # 按空格拆分行数据
    fields = split(line)
    if length(fields) == 4
        # 提取原子类型和坐标
        atom_type = fields[1]
        coordinates = parse.(Float64, fields[2:4])  # 转换为浮点数
        push!(atom_data, coordinates/8.05916)
    end
end
atomh2o=Vector{Atom}([])



cpc=[1,1,1]
para=getpara()
kb=para["kb"]
h=para["h"]
amuM=para["amuM"]
invlt=inv(lattice_vectors)

for i in 1:length(atom_data)
    if mod(i,3)==1
    push!(atomh2o,Atom(atom_data[i],15.9994*amuM))
    else
        push!(atomh2o,Atom(atom_data[i],1.008*amuM))
    end
end
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
inicell.atoms=deepcopy(atomh2o)
natom=length(inicell.atoms)



#calculate
calculate_gr=true
calculate_pr=true
gr=RadialDistribution(0.5,2.0,200)
grsequence=1
prsequence=1
pr=SpatialDistribution(inicell,200,200,1)


projectname="ICE_111_200K"
ensemble="NVT"
Ts=200.0 #K
Ps=0.000101325/P00 #[p]
dt=0.00025 #ps
t0=0.01
N=4
Tb=Ts
Pb=Ps
maxstep=45000
dumpsequence=1
dumpcellsequence=100
printsequence=100
beginsamplestep=5000
trajsequence=4
NVE2NVT=4800
TQ=10
TW=1000
cutCoulomb=4.0
ifcheckConvergence=true
interaction=TIP3P(water,cutCoulomb=cutCoulomb)
traji=Trajectory(beginsamplestep,maxstep,trajsequence,dt)
trajv=[deepcopy(traji) for i in 1:natom]
initT!(Ts,inicell)

println("initemp=$(cell_temp(inicell))")
println("inipressure=$(pressure_int(inicell,interaction))")
println(inicell.lattice_vectors)

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
println("\nensemble:$ensemble,N=$N\n")
##logfile
open("$basepath\\Config.txt", "w") do logfile
    write(logfile, "projectname=$projectname\n")
    write(logfile,"IntergrateMethod:RK3/PIMD,Interaction:TIP3P")
    write(logfile, "ensemble:$ensemble\n")           
    write(logfile, "$natom  atoms\n")
    write(logfile, "N:$N\n")    
    write(logfile, "t0:$t0\n")   
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
    write(logfile, "beginsamplestep=$beginsamplestep\n")
    write(logfile, "trajsequence=$trajsequence\n")
end

open("$basepath\\Log.txt", "w") do io
    jldopen("$basepath\\DumpCell.JLD2","w") do iojl
cell=deepcopy(inicell)


if N==1
if ensemble=="NVE"
    z=cell2z(cell)
elseif ensemble=="NVT"
    z=cell2z(cell,thermostat)
elseif ensemble=="NPT"
    z=cell2z(cell,thermostat,barostat)
elseif ensemble=="NVTLangevin"
    z=cell2z(cell)
elseif ensemble=="NVT2NVE"
    z=cell2z(cell,thermostat)
else
    throw(ArgumentError("ensemble not found"))
end
update_rmat!(cell)
update_fmat!(cell,interaction)
elseif (N>=4)&&(mod(N,2)==0)
    if ensemble!="NVT"
        println("Only NVT ensemble is supported for PIMD")
    end
    cell=map2bead(inicell,interaction,N,Ts,r=0.0)
else
    throw("Wrong Beads Number")
end

cgflag=true
for i in 1:maxstep
if N==1
    if ensemble=="NVT2NVE"
        if i<NVE2NVT
            RK3_step!(z,dt,cell,interaction,thermostat)
        else
            if cgflag
                T=cell_temp(cell)
                if (abs(T-Ts)<5.0)
                    println("\nchange NVT to NVE at step $i,T=$T\n")
                    z=cell2z(cell)
                    RK3_step!(z,dt,cell,interaction)
                    cgflag=false
                else
                    RK3_step!(z,dt,cell,interaction,thermostat)
                end
            else
                RK3_step!(z,dt,cell,interaction)
            end
        end

    elseif ensemble=="NVE"
        RK3_step!(z,dt,cell,interaction)
    elseif ensemble=="NVT"
        RK3_step!(z,dt,cell,interaction,thermostat)
    elseif ensemble=="NPT"
        RK3_step!(z,dt,cell,interaction,thermostat,barostat)
    elseif ensemble=="NVTLangevin"
        LangevinVerlet_step!(dt,cell,interaction,Ts,t0)
    end
elseif (N>=4)&&(mod(N,2)==0)
    pimdLangevinStep!(cell,dt,Ts,interaction,t0=t0)
end
# pint=pressure_int(cell,interaction)
T=cell_temp(cell)
Ek=cell_Ek(cell,interaction,Ts)
Ep=cell_energy(cell,interaction)
# Ek=0.0
# Ep=0.0
# println("step=$i",cell.atoms[1].position)
if mod(i,dumpsequence)==0
    writedlm(io, [i,Ek,Ep,Ek+Ep,T,barostat.V]')
end
if mod(i,printsequence)==0
    println("step: $i, Ek: $Ek Ep: $Ep,E:$(Ek+Ep),T:$T,Rt:$(thermostat.Pt),V:$(barostat.V)")
end
if mod(i,dumpcellsequence)==0
    write(iojl, "cell_$i", cell)
end
if i>beginsamplestep
    if calculate_gr
        if mod(i,grsequence)==0
            calculate_gr!(gr,cell)
        end
    end
    if calculate_pr  
        if mod(i,prsequence)==0
            calculate_pr!(pr,cell)
        end
    end
    if mod(i,trajsequence)==0
       fix_traj!(trajv,cell)
    end
end

# if mod(i,1)==0
#     figi=visualize_unitcell_atoms(cell,sizelist=visize)
#     save("test1water1/$(lpad(i,3,'0')).png",figi)
# end

end
end
end

jldopen("$basepath\\tr.JLD2","w") do jltr
    write(jltr,"tr",trajv)
end

if calculate_gr
    Normalize_gr!(gr)
    p=Plots.plot(gr.rm,gr.ngr,label="g(r)",xlabel="r",ylabel="g(r)",title="g(r) of N=$N,T=$Ts",lw=2,dpi=600)
    Plots.savefig(p,"$basepath\\gr.png")
    jldopen("$basepath\\DumpCellgr.JLD2","w") do jlgr
        write(jlgr,"gr",gr)
    end

    
end

if calculate_pr
    Normalize_pr!(pr)
    fig=Plots.heatmap(pr.xl.*lattice_constant,pr.yl.*lattice_constant,transpose(pr.npr[:,:,1]),color=:viridis,title="Spatial Distribution Beads=$N,T=$Ts K",xlabel="x/A",ylabel="y/A")
    savefig(fig,"$basepath\\SpatialDistribution.png")
    jldopen("$basepath\\SpatialDistribution.JLD2","w") do iojl
        write(iojl,"pr",pr)
    end
end