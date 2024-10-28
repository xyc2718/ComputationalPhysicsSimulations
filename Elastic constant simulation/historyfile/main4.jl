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
ljk=1.0
function lj(r::Float64)
    return ljk*4*(1/r^12-1/r^6)
end
function Flj(r::Vector{Float64})
    rn=norm(r)
    return ljk*24*(2/rn^14-1/rn^8)*r
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

Ts=1.0
Ps=100.0

fcell=initcell(Ps,Ts,atoms,interaction,cp=[5,5,5],Prg=[0.1,8.0])
println(pressure_int(fcell,interaction))
println(cell_temp(fcell))
println(fcell.lattice_vectors)

barostat=Barostat(Ps,0.1,fcell.Volume,0.0)
open("dump_555.txt", "w") do io
    jldopen("dumpcell_555.jld2", "w") do file
for step in 1:2

    if step%1==0
        z=[step,(pressure_int(fcell,interaction)),(fcell.Volume),(cell_temp(fcell)),(barostat.Pv),(barostat.V)]
        writedlm(io, z')
        println("step $step,P=$(z[2]),V=$(z[3]),T=$(z[4]),Rv=$(z[5])")
        write(file, "cell_$step", fcell)
    end



    # LA_step!(fcell,interaction,barostat,4,0.001,1.0,gamma0=0.01,gammav=0.01)
    T=Ts
        dt=0.001
        gamma0=0.5
        gammav=0.1
        dist=Normal(0.0,1.0)
   
        kb=1.0
        dist=Normal(0,1)
        
        Pe=barostat.Pe
        natom=length(fcell.atoms)
        Pint=pressure_int(fcell,interaction)
        Rv=barostat.Pv
        W=barostat.W
        V=barostat.V

        for i in 1:natom
            fi=cell_forcei(fcell,interaction,i)
            fcell.atoms[i].momentum+=fi*dt*0.5-0.5*gamma0*fcell.atoms[i].momentum*dt/fcell.atoms[i].mass+rand(dist,3)*sqrt(kb*T*gamma0*dt)
        end

        Rvdt2=Rv+(Pint-Pe)*dt/2+sqrt(kb*T*gammav*dt)*rand(dist)-gammav*Rv*dt/2/barostat.W

        Vdt2=V+Rvdt2*dt/2/W
        Vdt=Vdt2+Rvdt2*dt/2/W

        fcell.Volume=Vdt
        fcell.lattice_vectors=copy(fcell.lattice_vectors).*((Vdt/V)^(1/3))
        barostat.V=Vdt
        cp=fcell.copy
        invlt=inv(fcell.lattice_vectors)

        for i in 1:natom
            
            ri=fcell.lattice_vectors*fcell.atoms[i].position
            ridt=ri+(fcell.atoms[i].momentum*dt./fcell.atoms[i].mass)*(V/Vdt)^(2/3)
            ridt=ridt*(Vdt/V)^(1/3)
            fcell.atoms[i].momentum=fcell.atoms[i].momentum*(V/Vdt)^(1/3)
            fcell.atoms[i].position=invlt*ridt
            for j in 1:3
                fcell.atoms[i].position[j]=mod(fcell.atoms[i].position[j],cp[j])
            end
        end
        Rvdt=Rvdt2+(Pint-Pe)*dt/2+sqrt(kb*T*gammav*dt)*rand(dist)-gammav*Rvdt2*dt/2/W
        barostat.Pv=Rvdt

        for i in 1:natom
            fi=cell_forcei(fcell,interaction,i)
            # if norm(fi)*dt^2/fcell.atoms[i].mass>V^(1/3)
            #     throw("step $step, atom $i meet too large force $fi,please decrease dt")
            # end
            fcell.atoms[i].momentum+=fi*dt/2-0.5*gamma0*fcell.atoms[i].momentum*dt/fcell.atoms[i].mass+rand(dist,3)*sqrt(kb*T*gamma0*dt)
        end
        
        # fig=visualize_unitcell_atoms0(fcell)
        # display(fig)
        # readline()
        if step==1
            write(file, "cell_9", fcell)

        end

        
end
end
end        