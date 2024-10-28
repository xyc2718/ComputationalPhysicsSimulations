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

function update_cell0!(z::Vector{Float64},cell::UnitCell)
    natom=Int((length(z)-4)/6)
    rl=z[1:3*natom]
    pl=z[3*natom+3:3*natom+3*natom+3]
###这里是否需要修改体积还有待考虑，现在加了p会向-inf发散  
    v=z[3*natom+2]
    v0=cell.Volume
    if v<0
        println("v<0 at v=$v")
    end
    ap=((v/v0)^(1/3)) 
    ltm=cell.lattice_vectors.*ap
    cell.lattice_vectors=ltm
    cell.Volume=v
    a,b,c=cell.copy
    # z[1:3*natom].=z[1:3*natom].*ap
    # z[3*natom+3:3*natom+3*natom+3].=z[3*natom+3:3*natom+3*natom+3]
#####################
    for i in 1:natom
        ri=inv(cell.lattice_vectors)*rl[3*i-2:3*i]
        ri[1]=mod(ri[1],a)
        ri[2]=mod(ri[2],b)
        ri[3]=mod(ri[3],c)
        cell.atoms[i].position=ri
        cell.atoms[i].momentum=pl[3*i-2:3*i]
    end
end

ct=5.0
interaction = Interaction(lj, Flj, ct, 0.1)



open("data_pthv1.txt", "w") do io
open("data_pthv1_z.txt","w") do io1

Ts=1.0
Ps=0.1
inicell=initcell(Ps,Ts,atoms,interaction,cp=[3,3,3],Prg=[0.03,8])


dt=0.001



cell=deepcopy(inicell)

natom=length(cell.atoms)
Qs=3*natom*Ts*(30*dt)^2
Ws=3*natom*Ts*(1000*dt)^2
thermostat = Thermostat(Ts, Qs, 0.0, 0.0)
barostat=Barostat(Ps,Ws,cell.Volume,0.0)
pl=[]
Tl=[]

z=cell2z(cell,thermostat,barostat);
for i in 1:1000000
k1=Hz(z,cell,interaction,thermostat,barostat)
zr=z+(dt/2).*k1
update_cell0!(zr,cell)
k2=Hz(zr,cell,interaction,thermostat,barostat)
zr=z-dt.*k1+(2*dt).*k2
update_cell0!(zr,cell)
k3=Hz(zr,cell,interaction,thermostat,barostat)
z.=z+(dt/6).*(k1.+4*k2.+k3)
update_cell0!(z,cell)
pint=pressure_int(cell,interaction)
T=cell_temp(cell)
if mod(i,100)==0
println("step: ",i," Temp: ",T," Pressure: ",pint)
writedlm(io, [i, T, pint]')
writedlm(io1, z')
end

end
end
end