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

# 定义铜的晶格常数（单位：Å）
lattice_constant = 1.0

# 定义铜的FCC晶胞的基矢量
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

# 定义铜的FCC晶胞中的原子位置（单位：Å）
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

ct=5.0
interaction = Interaction(lj, Flj, ct, 0.1)

x=1:0.001:ct
y=interaction.cutenergy.(x)
lines(x,y)


function update_celli!(rl,pl,i::Int,cell::UnitCell)
    cell.atoms[i].position=inv(cell.lattice_vectors)*rl
    cell.atoms[i].momentum=pl
end

function LA_step!(fcell::UnitCell,interaction::Interaction,barostat::Barostat,n::Int,dt::Float64,T::Float64;gamma0::Float64=1.0,gammav::Float64=1.0)
kb=1.0
dist=Normal(0,1)

Pe=barostat.Pe
ddt=dt/n
natom=length(fcell.atoms)
pl=fill(zeros(3),(2*n+1,natom))
rl=fill(zeros(3),(2*n+1,natom))
Pint=pressure_int(fcell,interaction)
Rv=barostat.Pv


Rvdt2=Rv+(Pint-Pe)*dt/2+sqrt(kb*T*gammav*dt)*rand(dist)-gammav*Rv*dt/2/barostat.W



V0=fcell.Volume
Vdt2=V0+Rvdt2*dt/2/barostat.W
Vdt=Vdt2+Rvdt2*dt/2/barostat.W
# println("$Rv,$(Rvdt2*dt/2/barostat.W)")
mi=fcell.atoms[1].mass
fdt2=fill(zeros(3),natom)
fcell.Volume=Vdt
barostat.V=Vdt
fcell.lattice_vectors=fcell.lattice_vectors.*(Vdt/V0)^(1/3)
ltv=fcell.lattice_vectors


for i in 1:natom
    pl[1,i]=fcell.atoms[i].momentum
    rl[1,i]=ltv*fcell.atoms[i].position
end
for j in 1:n-1
    fl=fill(zeros(3),length(fcell.atoms))
    for i in 1:length(fcell.atoms)
        fi=cell_forcei(fcell,interaction,i)
        fl[i]=fi
    end
    for i in 1:length(fcell.atoms)
        if j==1
            pl[2*n+1,i]=pl[1,i].+(dt/2)*fl[i].+sqrt(kb*T*gamma0*dt)*rand(dist,3)-(gamma0*dt/2/mi)*pl[1,i]
        end
        pl[2*j,i]=pl[j,i]+(ddt/2).*fl[i]+sqrt(kb*T*gamma0*ddt)*rand(dist,3)-(gamma0*ddt/2/mi)*pl[j,i]
        rl[2*j+1,i]=rl[2*j-1,i]+(ddt/2).*pl[2*j,i]*ddt/mi
        update_celli!(rl[2*j+1,i],pl[2*j,i],i,fcell)
    end

    fl=fill(zeros(3),length(fcell.atoms))
    for i in 1:length(fcell.atoms)
        fi=cell_forcei(fcell,interaction,i)
        fl[i]=fi
    end

    for i in 1:length(fcell.atoms)
        pl[2*j+1,i]=pl[2*j,i]+ddt/2*fl[i]+sqrt(kb*T*gamma0*ddt)*rand(dist,3)-(gamma0*ddt/2/mi)*pl[2*j,i]
    end   
end
for i in 1:length(fcell.atoms)
    mi=fcell.atoms[i].mass
    rl[2*n+1,i]=rl[2*n-1,i]+((V0/Vdt2)^(2/3)/mi*ddt).*pl[2*n,i]
    rl[2*n+1,i]=rl[2*n+1,i].*((Vdt/V0)^(1/3))
    pl[n+1,i]=pl[n+1,i].*((V0/Vdt)^(1/3))
end

invlt=inv(fcell.lattice_vectors)
for i in 1:length(fcell.atoms)
    fcell.atoms[i].position=invlt*rl[2*n+1,i]
end

for i in 1:length(fcell.atoms)
    fi=cell_forcei(fcell,interaction,i)
    pl[2*n+1,i]=pl[n+1,i]+(dt/2).*fi-(gamma0*dt/2/mi)*pl[n+1,i]+sqrt(kb*T*gammav*dt)*rand(dist,3)
end

for i in 1:length(fcell.atoms)
    fcell.atoms[i].momentum=pl[2*n+1,i]
end

Pint=pressure_int(fcell,interaction)
Rvdt=Rvdt2+(Pint-Pe).*(dt/2)+sqrt(kb*T*gammav*dt)*rand(dist)-gammav*Rvdt2*dt/2/barostat.W
barostat.Pv=Rvdt
end

Ts=100.0
Ps=100.0

fcell=initcell(Ps,Ts,atoms,interaction,cp=[3,3,3],Prg=[0.03,8])
println(pressure_int(fcell,interaction))
println(cell_temp(fcell))


barostat=Barostat(Ps,0.01,fcell.Volume,0.0)
open("dump.txt", "w") do io
for i in 1:100000
    LA_step!(fcell,interaction,barostat,4,0.001,1.0,gamma0=0.01,gammav=0.1)
    if i%50==0
        z=[i,(pressure_int(fcell,interaction)),(fcell.Volume),(cell_temp(fcell)),(barostat.Pv)]
        writedlm(io, z')
        println("step $i,P=$(z[2]),V=$(z[3]),T=$(z[4]),Rv=$(z[5])")
    end
end
end        