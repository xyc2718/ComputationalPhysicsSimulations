"""
@author:XYC
@email:22307110070@m.fudan.edu.cn
MD for NPT ensemble,include:
Andersen Nose-Hoover method:
RK3_step!: Intergrate by 3 order Runge Kuta
Andersen_Hoover_NPT_step!:Intergrate by Liouville operator and Tort Decomposition[1]

Andersen Langvin method:
LA_step!

NVT ensemble:
    Nose-Hoover method:Intergrate by 3 order Runge Kuta
NVE ensemble:Integrate by 3 order Runge Kuta 

Reference:
[1]Jalkanen, J., & Müser, M. H. (2015). Systematic analysis and modification of embedded-atom potentials: Case study of copper. Modelling and Simulation in Materials Science and Engineering, 23(7), 074001. https://doi.org/10.1088/0965-0393/23/7/074001
[2]Bereau, T. (2015). Multi-timestep Integrator for the Modified Andersen Barostat. Physics Procedia, 68, 7–15. https://doi.org/10.1016/j.phpro.2015.07.101
"""
module MD
using Distributions
using StaticArrays
using LinearAlgebra
using GLMakie 
using ..Model
using Base.Threads
using JLD2


export pressure_int,Thermostat,Barostat,Hz,RK3_step!,z2atoms,z2cell,cell2z,dUdV_default,update_cell!,initT!,initcell,dUdV_default,Nhcpisoint!,Andersen_Hoover_NPT_step!,LA_step!,minEenergyCell,provide_cell,LangevinVerlet_step!


"""
default function for dUdV_default
向前向后微分数值计算dU/dV;步长默认取10e-9Volume,这一项在NPT恒压中是重要的,忽略会导致恒压不稳定
:param inicell::UnitCell
"""
function dUdV_default(inicell::UnitCell,interaction::AbstractInteraction,dV::BigFloat=BigFloat("1e-9"))
    # jldopen("testcell.txt","w") do iojl
    natom=length(inicell.atoms)
    V0::BigFloat=inicell.Volume
    dV0::BigFloat=inicell.Volume*dV
    V1=V0+dV0
    V2=V0-dV0
    ltv=inicell.lattice_vectors
    ltv1=ltv*(V1/V0)^(1/3)
    ltv2=ltv*(V2/V0)^(1/3)
    cp=inicell.copy

    dcell=deepcopy(inicell)
    dcell.lattice_vectors.=ltv1
    # for i in 1:natom
    #     ri=(inv(ltv1))*ltv*dcell.atoms[i].position
    #     for j in 1:3
    #         dcell.atoms[i].position[j]=mod(ri[j]+cp[j],2*cp[j])-cp[j]
    #     end
    #     # dcell.atoms[i].position=(ltv1)*ltv*dcell.atoms[i].position
    # end
    apply_PBC!(dcell)
    update_rmat!(dcell)
    energy1=cell_energy(dcell,interaction)
    # fig=visualize_unitcell_atoms0(dcell)
    # display(fig)

    dcell=deepcopy(inicell)
    dcell.lattice_vectors.=ltv2
    # for i in 1:natom
    #     ri=(inv(ltv2))*ltv*dcell.atoms[i].position
    #     for j in 1:3
    #         dcell.atoms[i].position[j]=mod(ri[j]+cp[j],2*cp[j])-cp[j]
    #     end
    #     # dcell.atoms[i].position=(ltv2)*ltv*dcell.atoms[i].position
    # end
    apply_PBC!(dcell)
    update_rmat!(dcell)
    energy2=cell_energy(dcell,interaction)
    # println("E1=$energy1,E2=$energy2")

    # kk=0.0
    # if energy2>1e10
        
    #     kk+=1.0
    #     write(iojl, "cell_$kk", dcell)

    # end
  return (energy1-energy2)/dV0/2
end



"""
等效压强Pint
:param cell: UnitCell
:param interaction: Interaction
:param dUdV: function for dUdV(UnitCell,interaction)
return Pint
"""
function pressure_int(cell::UnitCell,interaction::AbstractInteraction,dUdV::T=dUdV_default) where T
    v=cell.Volume
    Pint1=0.0
    Pint2=0.0
    lt=cell.lattice_vectors
    for i in 1:length(cell.atoms)
        atom=cell.atoms[i]
        pm=atom.momentum
        ri=lt*atom.position
        fi=cell_forcei(cell,interaction,i)
        Pint1+=dot(pm,pm)/atom.mass
        Pint2+=dot(ri,fi)
    end
    # println("Pint1=$Pint1,Pint2=$Pint2")
    Pint=Pint1+Pint2
    return Pint/v/3-dUdV(cell,interaction)
end

"""
等效压强Pint
:param cell: UnitCell
:param interaction: Interaction
:param dUdV: function for dUdV(UnitCell,interaction)
return Pint
"""
function pressure_int_rfl(cell::UnitCell,interaction::AbstractInteraction,dUdV::T=dUdV_default) where T
    v=cell.Volume
    Pint1=0.0
    Pint2=0.0
    lt=cell.lattice_vectors
    natom=length(cell.atoms)
    fl = Vector{SVector{3, Float64}}(undef, natom)
    for i in 1:length(cell.atoms)
        atom=cell.atoms[i]
        pm=atom.momentum
        ri=lt*atom.position
        fi=cell_forcei(cell,interaction,i)
        fl[i]=fi
        Pint1+=dot(pm,pm)/atom.mass
        Pint2+=dot(ri,fi)
    end
    # println("Pint1=$Pint1,Pint2=$Pint2")
    Pint=Pint1+Pint2
    return (Pint/v/3-dUdV(cell,interaction),fl)
end

"""
恒温器
:param T: 目标温度
:param Q: 热浴质量
:param Rt: 热浴变量 
:param Pt: 热浴动量
"""
mutable struct Thermostat
    T::Float64  # 目标温度
    Q::Float64  # 热浴质量
    Rt::Float64  # 热浴变量 
    Pt::Float64  # 热浴动量
end

"""
恒压器
:param Pe: 目标压力
:param V: 系统体积
:param W: 压力浴质量
:param Pv: 压力浴动量
"""
mutable struct Barostat
    Pe::Float64  # 目标压力
    W::Float64  # 压力浴质量
    V::Float64  # 系统体积
    Pv::Float64  # 压力浴动量 logVe or Ve depend on situation
end

function provide_cell(cell::UnitCell,dt::Float64)
    cp=cell.copy
    cpl=cell.lattice_vectors*cp
    for atom in cell.atoms
        vdt=atom.momentum/atom.mass*dt*2
        if any(vdt.>cpl)
            println("Error!The Vdt>the length of Box with atoms:$atom and Vdt:$(vdt)")
        end
    end
end
function provide_cell(bdc::BeadCell,dt::Float64)
    for cell in bdc.cells
    cp=cell.copy
    cpl=cell.lattice_vectors*cp
    for atom in cell.atoms
        vdt=atom.momentum/atom.mass*dt*2
        if any(vdt.>cpl)
            println("Error!The Vdt>the length of Box with atoms:$atom and Vdt:$(vdt)")
        end
    end
end
end
"""
计算dH/dz
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param cell: UnitCell
:param interaction: Interaction
:param thermostat: Thermostat
:param barostat: Barostat
:param dUdV: function for dUdV
"""
function Hz(z::Vector{Float64},cell::UnitCell,interaction::AbstractInteraction,thermostat::Thermostat,barostat::Barostat;dUdV::T=dUdV_default) where T
    kb=8.617332385e-5 #eV/K
    dim=3*length(cell.atoms)+2
    Hz=zeros(dim*2)
    natom=length(cell.atoms)
    if 2*dim!=length(z)
        throw("The dimension of z is not consist with the dimension of the system. z should be natom*3+2")
    end
    W=barostat.W
    Q=thermostat.Q
    temp=thermostat.T
    Pe=barostat.Pe
    Pint=pressure_int(cell,interaction,dUdV)
    addp=sum([dot(atom.momentum,atom.momentum)/atom.mass for atom in cell.atoms])
    # @threads 
    for i in 1:natom
        atom=cell.atoms[i]
        mi=atom.mass
        Hz[3*i-2]=z[3*i-2+dim]/mi+z[2*dim]*z[3*i-2]/W
        Hz[3*i-1]=z[3*i-1+dim]/mi+z[2*dim]*z[3*i-1]/W
        Hz[3*i]=z[3*i+dim]/mi+z[2*dim]*z[3*i]/W
        fi=cell_forcei(cell,interaction,i)
        Hz[dim+3*i-2]=fi[1]-(1+1/natom)*z[2*dim]/W*z[dim+3*i-2]-z[2*dim-1]*z[dim+3*i-2]/Q
        Hz[dim+3*i-1]=fi[2]-(1+1/natom)*z[2*dim]/W*z[dim+3*i-1]-z[2*dim-1]*z[dim+3*i-1]/Q
        Hz[dim+3*i]=fi[3]-(1+1/natom)*z[2*dim]/W*z[dim+3*i]-z[2*dim-1]*z[dim+3*i]/Q
    end
    Hz[dim]=3*z[dim]*z[2*dim]/W
    Hz[2*dim]=3*z[dim]*(Pint-Pe)+1/natom*addp-z[2*dim-1]*z[2*dim]/Q
    Hz[dim-1]=z[2*dim-1]/Q
    Hz[2*dim-1]=addp+z[2*dim]^2/W-(3*natom+1)*kb*temp
    return Hz
end





"""
计算dH/dz in NVT
:param z: z=[r1...rn,Rt,p1,...pn,Pt]
:param cell: UnitCell
:param interaction: Interaction
:param thermostat: Thermostat
"""
function Hz(z::Vector{Float64},cell::UnitCell,interaction::AbstractInteraction,thermostat::Thermostat)
    kb=8.617332385e-5 #eV/K
    dim=3*length(cell.atoms)+1
    Hz=zeros(dim*2)
    natom=length(cell.atoms)
    if 2*dim!=length(z)
        throw("The dimension of z is not consist with the dimension of the system. z should be natom*3+1")
    end
    Q=thermostat.Q
    temp=thermostat.T
    addp=sum([dot(atom.momentum,atom.momentum)/atom.mass for atom in cell.atoms])
    # @threads 
    for i in 1:natom
        atom=cell.atoms[i]
        mi=atom.mass
        Hz[3*i-2]=z[3*i-2+dim]/mi
        Hz[3*i-1]=z[3*i-1+dim]/mi
        Hz[3*i]=z[3*i+dim]/mi
        fi=cell_forcei(cell,interaction,i)
        Hz[dim+3*i-2]=fi[1]-z[2*dim]*z[dim+3*i-2]/Q
        Hz[dim+3*i-1]=fi[2]-z[2*dim]*z[dim+3*i-1]/Q
        Hz[dim+3*i]=fi[3]-z[2*dim]*z[dim+3*i]/Q
    end
    Hz[dim]=z[2*dim]/Q
    Hz[2*dim]=addp-3*natom*kb*temp
    # println("addp:",addp)
    # println(3*natom*kb*temp)
    return Hz
end

"""
Hz for NVT with Langevin Thermostat
"""
function Hz(z::Vector{Float64},cell::UnitCell,interaction::AbstractInteraction,Ts::Float64,t0::Float64,dt::Float64)
    kb=8.617332385e-5 #eV/K
    dim=3*length(cell.atoms)
    Hz=zeros(dim*2)
    natom=length(cell.atoms)
    if 2*dim!=length(z)
        throw("The dimension of z is not consist with the dimension of the system. z should be natom*3")
    end
    # @threads 
    for i in 1:natom
        atom=cell.atoms[i]
        mi=atom.mass
        sigma =2*sqrt(mi*kb*Ts/t0/dt)
        Hz[3*i-2]=z[3*i-2+dim]/mi
        Hz[3*i-1]=z[3*i-1+dim]/mi
        Hz[3*i]=z[3*i+dim]/mi
        fi=cell_forcei(cell,interaction,i)
        Hz[dim+3*i-2]=fi[1]-z[dim+3*i-2]/t0+sigma*randn()
        Hz[dim+3*i-1]=fi[2]-z[dim+3*i-1]/t0+sigma*randn()
        Hz[dim+3*i]=fi[3]-z[dim+3*i]/t0+sigma*randn()
    end
    return Hz
end

"""
Hz for NVE
"""
function Hz(z::Vector{Float64},cell::UnitCell,interaction::AbstractInteraction)
    kb=8.617332385e-5 #eV/K
    dim=3*length(cell.atoms)
    Hz=zeros(dim*2)
    natom=length(cell.atoms)
    if 2*dim!=length(z)
        throw("The dimension of z is not consist with the dimension of the system. z should be natom*3")
    end
    # @threads 
    for i in 1:natom
        atom=cell.atoms[i]
        mi=atom.mass
        Hz[3*i-2]=z[3*i-2+dim]/mi
        Hz[3*i-1]=z[3*i-1+dim]/mi
        Hz[3*i]=z[3*i+dim]/mi
        fi=cell_forcei(cell,interaction,i)
        Hz[dim+3*i-2]=fi[1]
        Hz[dim+3*i-1]=fi[2]
        Hz[dim+3*i]=fi[3]
    end
    return Hz
end

"""
RK3 for NVE
"""
function RK3_step!(z::Vector{Float64},dt::Float64,cell::UnitCell, interaction::AbstractInteraction)
    dim=Int(length(z)/2)
    k1=Hz(z,cell,interaction)
    zr=z+(dt/2).*k1
    update_cell_NVE!(zr,cell,interaction)
    updatezr!(zr,cell)
    k2=Hz(zr,cell,interaction)
    zr.=z-dt.*k1+(2*dt).*k2
    update_cell_NVE!(zr,cell,interaction)
    updatezr!(zr,cell)
    k3=Hz(zr,cell,interaction)
    z.=z+(dt/6).*(k1.+4*k2.+k3)
    update_cell_NVE!(z,cell,interaction)
    updatezr!(z,cell)
end 

"""
RK3 for NVT with Langevin thermostat(可靠性存疑)
"""
function RK3_step!(z::Vector{Float64},dt::Float64,cell::UnitCell, interaction::AbstractInteraction,Ts::Float64,t0::Float64)
    dim=Int(length(z)/2)
    k1=Hz(z,cell,interaction,Ts,t0,dt)
    zr=z+(dt/2).*k1
    update_cell_NVE!(zr,cell,interaction)
    updatezr!(zr,cell)
    k2=Hz(zr,cell,interaction,Ts,t0,dt)
    zr.=z-dt.*k1+(2*dt).*k2
    update_cell_NVE!(zr,cell,interaction)
    updatezr!(zr,cell)
    k3=Hz(zr,cell,interaction,Ts,t0,dt)
    z.=z+(dt/6).*(k1.+4*k2.+k3)
    update_cell_NVE!(z,cell,interaction)
    updatezr!(z,cell)
end 

function LangevinVerlet_step!(dt::Float64,cell::UnitCell, interaction::AbstractInteraction,Ts::Float64,t0::Float64)
    para=getpara()
    kb=para["kb"]
    invlt=inv(cell.lattice_vectors)
    c1=exp(-dt/2/t0)
    c2=sqrt(1-c1^2)
    for atom in cell.atoms
        mi=atom.mass
        atom.momentum=c1*atom.momentum+c2*randn(3).*sqrt(mi*kb*Ts)
    end
    for i in eachindex(cell.atoms)
        atom=cell.atoms[i]
        mi=atom.mass
        fi=cell_forcei(cell,interaction,i)
        atom.position+=invlt*(atom.momentum*dt/mi+fi*dt^2/2/mi)
        atom.momentum+=fi*dt/2
    end
    apply_PBC!(cell)
    update_rmat!(cell)
    update_fmat!(cell,interaction)

    for i in eachindex(cell.atoms)
        atom=cell.atoms[i]
        fi=cell_forcei(cell,interaction,i)
        atom.momentum+=fi*dt/2
    end
    for atom in cell.atoms
        mi=atom.mass
        atom.momentum=c1*atom.momentum+c2*randn(3).*sqrt(mi*kb*Ts)
    end

end 


"""
RK3步进,NVT
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param dt: 步长
:param cell: UnitCell
:param interaction: Interaction
:param thermostat: Thermostat
return z
"""
function RK3_step!(z::Vector{Float64},dt::Float64,cell::UnitCell, interaction::AbstractInteraction,thermostat::Thermostat)
    dim=Int(length(z)/2)
    k1=Hz(z,cell,interaction,thermostat)
    zr=z+(dt/2).*k1
    update_cell_NVT!(zr,cell,interaction)
    updatezr!(zr,cell)
    k2=Hz(zr,cell,interaction,thermostat)
    zr.=z-dt.*k1+(2*dt).*k2
    update_cell_NVT!(zr,cell,interaction)
    updatezr!(zr,cell)
    k3=Hz(zr,cell,interaction,thermostat)
    z.=z+(dt/6).*(k1.+4*k2.+k3)
    update_cell_NVT!(z,cell,interaction)
    updatezr!(z,cell)
    thermostat.Pt=z[2*dim]
    thermostat.Rt=z[dim]
end




"""
RK3步进
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param dt: 步长
:param cell: UnitCell
:param interaction: Interaction
:param thermostat: Thermostat
:param barostat: Barostat
:param dUdV: function for dUdV
return z
"""
function RK3_step!(z::Vector{Float64},dt::Float64,cell::UnitCell, interaction::AbstractInteraction, thermostat::Thermostat, barostat::Barostat)
    k1=Hz(z,cell,interaction,thermostat,barostat)
    zr=z+(dt/2).*k1
    update_cell!(zr,cell,interaction)
    updatezr!(zr,cell)
    k2=Hz(zr,cell,interaction,thermostat,barostat)
    zr.=z-dt.*k1+(2*dt).*k2
    update_cell!(zr,cell,interaction)
    updatezr!(zr,cell)
    k3=Hz(zr,cell,interaction,thermostat,barostat)
    z.=z+(dt/6).*(k1.+4*k2.+k3)
    update_cell!(z,cell,interaction)
    updatezr!(z,cell)
    dim=Int(length(z)/2)
    thermostat.Pt=z[2*dim-1]
    barostat.Pv=z[2*dim]
    thermostat.Rt=z[dim-1]
    barostat.V=z[dim]
    end


"""
根据将z转化为atoms
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:return atoms
"""
function z2atoms(z::Vector{Float64},cell::UnitCell)
    natom=Int((length(z)-4)/6)
    rl=z[1:3*natom]
    pl=z[3*natom+3:3*natom+3*natom+3]
    atoms=Vector{Atom}(undef,natom)
    for i in 1:natom
        atom=Atom(inv(cell.lattice_vectors)*rl[3*i-2:3*i],pl[3*i-2:3*i],cell.atoms[i].mass,cell.atoms[i].cn,cell.atoms[i].bound)
        atoms[i]=atom
    end
    return atoms
end


"""
将z转化为UnitCell
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param cell: UnitCell
:return newcell
"""
function z2cell(z::Vector{Float64},cell::UnitCell)
    atoms=z2atoms(z,cell)
    newcell=UnitCell(cell.lattice_vectors,atoms,cell.copy)
    return newcell
end


"""
根据z修改cell,直接修改原始cell,z
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param cell: UnitCell
"""
function update_cell!(z::Vector{Float64},cell::UnitCell,interaction::AbstractInteraction)
    natom=Int((length(z)-4)/6)
    rl=z[1:3*natom]
    pl=z[3*natom+3:6*natom+2]
    v=z[3*natom+2]
    v0=cell.Volume
    if v<0
        println("v<0 at v=$v")
    end
    ap=((v/v0)^(1/3)) 
    ltm=cell.lattice_vectors*ap  
    cell.lattice_vectors=ltm
    cell.Volume=v
    invltv=inv(ltm)
    for i in 1:natom
        ri=invltv*rl[3*i-2:3*i]
        cell.atoms[i].position=ri
        cell.atoms[i].momentum=pl[3*i-2:3*i]
    end
    apply_PBC!(cell)
    update_rmat!(cell)
    update_fmat!(cell,interaction)
end




"""
根据z修改cell,直接修改原始cell,z
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param cell: UnitCell
"""
function update_cell_NVT!(z::Vector{Float64},cell::UnitCell,interaction::AbstractInteraction)
    natom=Int((length(z)-2)/6)
    rl=z[1:3*natom]
    pl=z[3*natom+2:6*natom+1]
    invltv=inv(cell.lattice_vectors)
    for i in 1:natom
        ri=invltv*rl[3*i-2:3*i]
        cell.atoms[i].position=ri
        cell.atoms[i].momentum=pl[3*i-2:3*i]
    end
   
    apply_PBC!(cell)
    update_rmat!(cell)
    update_fmat!(cell,interaction)
end


function update_cell_NVE!(z::Vector{Float64},cell::UnitCell,interaction::AbstractInteraction)
    natom=Int((length(z))/6)
    rl=z[1:3*natom]
    pl=z[3*natom+1:6*natom]
    a,b,c=cell.copy
    invltv=inv(cell.lattice_vectors)
    for i in 1:natom
        ri=invltv*rl[3*i-2:3*i]
        cell.atoms[i].position=ri
        cell.atoms[i].momentum=pl[3*i-2:3*i]
    end
   
    update_rmat!(cell)
    update_fmat!(cell,interaction)
end



"""
根据rl,pl修改cell的i原子,直接修改cell
"""
function update_celli!(rl,pl,i::Int,cell::UnitCell)
    cell.atoms[i].position=inv(cell.lattice_vectors)*rl
    cell.atoms[i].momentum=pl
    # update_rmati!(cell,i)
    # update_fmat!(cell,interaction)
end


"""
将UnitCell转化为z
:param cell: UnitCell
:param thermostat: Thermostat
:param barostat: Barostat
return z
"""
function cell2z(cell::UnitCell,thermostat::Thermostat,barostat::Barostat)
    rl=[cell.lattice_vectors*atom.position for atom in cell.atoms];
    pl=[atom.momentum for atom in cell.atoms];
    z=vcat(vcat(rl...),thermostat.Rt,barostat.V,vcat(pl...),thermostat.Pt,barostat.Pv);
    return z
end

"""
将UnitCell转化为z for NVT
:param cell: UnitCell
:param thermostat: Thermostat
return z
"""
function cell2z(cell::UnitCell,thermostat::Thermostat)
    rl=[cell.lattice_vectors*atom.position for atom in cell.atoms];
    pl=[atom.momentum for atom in cell.atoms];
    z=vcat(vcat(rl...),thermostat.Rt,vcat(pl...),thermostat.Pt);
    return z
end


function cell2z(cell::UnitCell)
    rl=[cell.lattice_vectors*atom.position for atom in cell.atoms];
    pl=[atom.momentum for atom in cell.atoms];
    z=vcat(vcat(rl...),vcat(pl...));
    return z
end

"""
根据cell修改z中的r
"""
function updatezr!(z,cell::UnitCell)    
    ltv=cell.lattice_vectors
    for i in  eachindex(cell.atoms)
        z[3*i-2:3*i].=ltv*cell.atoms[i].position
    end
end


function initT!(T::Float64,cell::UnitCell) 
    m=cell.atoms[1].mass
    kb=8.617332385e-5 #eV/K
    natom=length(cell.atoms)
    sigma=sqrt(kb*T/m)
    Ekstd=3*natom*kb*T/2
    for i in 1:natom
        cell.atoms[i].momentum = rand(Normal(0,sigma),3) 
    end
    Ek=0.0
    for i in 1:natom
        Ek+=0.5*norm(cell.atoms[i].momentum)^2/m
    end
    pc=zeros(3)
    for i in 1:natom
        pc.+=cell.atoms[i].momentum
    end
    pc./=natom
    for i in 1:natom
        cell.atoms[i].momentum.=cell.atoms[i].momentum.-pc
    end

    for i in 1:natom
        cell.atoms[i].momentum=cell.atoms[i].momentum.*sqrt(Ekstd/Ek)
    end
end

function minEenergyCell(T::Float64,atoms::Vector{Atom},interaction::AbstractInteraction,cp::Vector{Int};rg=[0.5,8.0],n::Int=100)
    cl = range(rg[1], stop=rg[2], length=n)
    El=Vector{Float64}([])
    for lt in cl
        lattice_vectors=collect((Matrix([
            lt 0.0 0.0; #a1
            0.0 lt 0.0; #a2
            0.0 0.0 lt] #a3
        ))')
        cell=UnitCell(lattice_vectors,atoms)
        cpcell=copycell(cell,cp...)
        fcell=filtercell(cpcell)
        update_rmat!(fcell)
        Ei=cell_energy(fcell,interaction)
        push!(El,Ei)
    end

    minindex=argmin(El)
    lt=cl[minindex]
    lattice_vectors=collect((Matrix([
        lt 0.0 0.0; #a1
        0.0 lt 0.0; #a2
        0.0 0.0 lt] #a3
    ))')
    cell=UnitCell(lattice_vectors,atoms)
    cpcell=copycell(cell,cp...)
    fcell=filtercell(cpcell)
    initT!(T,fcell)
    update_rmat!(fcell)
    update_fmat!(fcell,interaction)
    return fcell

end


function initcell(P::Float64,T::Float64,atoms::Vector{Atom},interaction::AbstractInteraction;cp::Vector{Int}=[1,1,1],Prg::Vector{Float64}=[0.5,6.0],n=100)
    cl = range(Prg[1], stop=Prg[2], length=n)
    for lt in cl
        lattice_vectors = (Matrix([
                            lt 0.0 0.0; #a1
                            0.0 lt 0.0; #a2
                            0.0 0.0 lt] #a3
                        ))
        cell=UnitCell(lattice_vectors,atoms)
        cpcell=copycell(cell,cp...)
        fcell=filtercell(cpcell)
        initT!(T,fcell)
        update_rmat!(fcell)
        pa=pressure_int(fcell,interaction)-P
        if abs(pa)<1e-2
            update_rmat!(fcell)
            return fcell
        end
    end
    throw("warning Can't find the right lattice constant for P=$P,T=$T in Range $Prg for $n points")
end



"""
对于Nosehoover链的NHC演化算符,采用nresn*3的多部微分,将会直接修改cell.posi,momentum,thermostat.Pt,Rt,barostat.Pv see Andersen_Hoover_NPT_step! for more information
:param cell: UnitCell
:param interaction: Interaction
:param thermostatchain: Vector{Thermostat}
:param barostat: Barostat
:param dt: Float64
:param nresn: Float64
"""
function Nhcpisoint!(cell::UnitCell,interaction::AbstractInteraction,thermostatchain::Vector{Thermostat},barostat::Barostat,dt::Float64;nresn::Int=3)
    kb=8.617332385e-5 #eV/K
    natom=length(cell.atoms)
    Nf=3*natom ##自由度
    T=thermostatchain[1].T
    GN1KT=(Nf)*kb*T 
    GKT=kb*T
    odnf=1+3/Nf
    W=barostat.W
    V=cell.Volume
    Pe=barostat.Pe
    Pint=pressure_int(cell,interaction)
    nnos=length(thermostatchain) 
    glogs=zeros(nnos)
    vlogs=[th.Pt for th in thermostatchain] ##恒温器动量
    xlogs=[th.Rt for th in thermostatchain] ##恒温器位置
    vlogv=barostat.Pv ##压力浴动量 v_epsilon epsilon=1/3 log(V)
    glogv=0.0  #G_eps
    
    """
    nys*3差分 nresn->nc,第一次多步
    nyosh->nys 第二维多步
    其中(nys,wj)表格见:Yoshida, H. (1990). Construction of higher order symplectic integrators. Physics Letters A, 150(5–7), 262–268. https://doi.org/10.1016/0375-9601(90)90092-3
    """
    nyosh=3
    w1=1/(2-2^(1/3)) 
    w3=w1
    w2=1-2*w1
    wdti=[w1,w2,w3]*dt
    wdti2=wdti./2/nresn
    wdti4=wdti./4/nresn
    wdti8=wdti./8/nresn
    scale::BigFloat=1.0
    kint=0.0
    for i in 1:natom 
        kint=kint+dot(cell.atoms[i].momentum,cell.atoms[i].momentum)/cell.atoms[i].mass
    end
 
        glogs[1]=(kint+W*vlogv^2-GN1KT)/thermostatchain[1].Q
        glogv=(odnf*kint+3.0*(Pint-Pe)*V)/W

    for iresn in 1:nresn 
        for iyosh in 1 :nyosh
                   
            vlogs[nnos]=vlogs[nnos]+glogs[nnos]*wdti4[iyosh]
            
            for inos in 1:nnos-1
                AA=exp(-wdti8[iyosh]*vlogs[nnos+1-inos])
                vlogs[nnos-inos]=vlogs[nnos-inos]*AA^2+wdti4[iyosh]*glogs[nnos-inos]*AA
            end
            AA=exp(-wdti8[iyosh]*vlogs[1])
            vlogv=vlogv*AA^2+wdti4[iyosh]*glogv*AA

            AA=exp(-wdti2[iyosh]*(vlogs[1]+odnf*vlogv))
            scale=scale*AA
            kint=kint*AA^2
            glogv=(odnf*kint+3.0*(Pint-Pe)*V)/W
            
    
            for inos in 1:nnos
                xlogs[inos]=xlogs[inos]+vlogs[inos]*wdti2[iyosh]
            end
            AA=exp(-wdti8[iyosh]*vlogs[1])
            vlogv=vlogv*AA^2+wdti4[iyosh]*glogv*AA
            glogs[1]=(kint+W*vlogv^2-GN1KT)/thermostatchain[1].Q
            
            for inos in 1: nnos-1
                AA=exp(-wdti8[iyosh]*vlogs[inos+1])
                vlogs[inos]=vlogs[inos]*AA*AA+wdti4[iyosh]*glogs[inos]*AA
                glogs[inos+1]=(thermostatchain[inos].Q*vlogs[inos]*vlogs[inos]-GKT)/thermostatchain[inos+1].Q
            end
            vlogs[nnos]=vlogs[nnos]+glogs[nnos]*wdti4[iyosh]
        end

    end
    for i in 1:natom 
        cell.atoms[i].momentum=cell.atoms[i].momentum*scale
    end
    barostat.Pv=vlogv
    for inos in 1:nnos 
        thermostatchain[inos].Rt=xlogs[inos]
        thermostatchain[inos].Pt=vlogs[inos]
    end 
end


"""
Andersen-Hoover NPT的演化算符,L_NHCP的tort分解为L_NHC(dt/2) L1 L2 L1 L_NHC(dt/2),其中L_NHC部分由Nhcpisoint!完成,使用nresn*3的多部演化
直接修改cell.posi,momentum,Volume,lattice_vectors,thermostat.Pt,Rt,barostat.Pv,V
:param cell: UnitCell
:param interaction: Interaction 
:param thermostatchain: Vector{Thermostat},Nose Hoover链
:param barostat: Barostat 恒压器
:param dt: Float64
:param nresn: Int=3

Reference:
Jalkanen, J., & Müser, M. H. (2015). Systematic analysis and modification of embedded-atom potentials: Case study of copper. Modelling and Simulation in Materials Science and Engineering, 23(7), 074001. https://doi.org/10.1088/0965-0393/23/7/074001
"""
function Andersen_Hoover_NPT_step!(cell::UnitCell,interaction::AbstractInteraction,thermostatchain::Vector{Thermostat},barostat::Barostat,dt::Float64;nresn::Int=3)
    dt2=dt/2
    I::BigFloat= 1.0    #使用级数计算sinh(dt)/dt 
    E2 = I / 6.0
    E4 = E2 / 20.0
    E6 = E4 / 42.0
    E8 = E6 / 72.0
    natom=length(cell.atoms)
    Nhcpisoint!(cell,interaction,thermostatchain,barostat,dt,nresn=nresn)
    xlogv=1/3*log(barostat.V)
    vlogv=barostat.Pv
    initial_volume=cell.Volume
    for i in 1:natom
        fi=cell_forcei(cell,interaction,i)
        cell.atoms[i].momentum.+=dt2*fi
    end
    AA=exp(dt2*vlogv)
    AA2=AA*AA
    arg2=(vlogv*dt2)^2
    poly=(((E8*arg2+E6)*arg2+E4)*arg2+E2)+I
    BB=AA*poly*dt
    invltv=inv(cell.lattice_vectors)
    for i in 1:natom
        cell.atoms[i].position.=cell.atoms[i].position*AA2+(BB)*(invltv*cell.atoms[i].momentum/cell.atoms[i].mass)
    end
    xlogv=xlogv+vlogv*dt
    cell.Volume=exp(3*xlogv)
    cell.lattice_vectors.=cell.lattice_vectors*(cell.Volume/initial_volume)^(1/3)

    update_rmat!(cell)
    update_fmat!(cell,interaction)

    barostat.V=cell.Volume
    barostat.Pv=vlogv
    for i in 1:natom
        fi=cell_forcei(cell,interaction,i)
        cell.atoms[i].momentum.+=dt2*fi
    end
    Nhcpisoint!(cell,interaction,thermostatchain,barostat,dt,nresn=nresn)
    apply_PBC!(cell)
    update_rmat!(cell)
    update_fmat!(cell,interaction)
    
end

"""
Langvin Anderson方式控制NPT
:parm fcell: UnitCell
:parm interaction: Interaction
:parm dt: Float64
:parm T: Float64
:parm barostat: Barostat
:parm gamma0: Float64
:parm gammav: Float64
:parm n: Int=4
this is not surport for molecule with PBC yet
Ref:Bereau, T. (2015). Multi-timestep Integrator for the Modified Andersen Barostat. Physics Procedia, 68, 7–15. https://doi.org/10.1016/j.phpro.2015.07.101
"""
function LA_step!(fcell::UnitCell,interaction::AbstractInteraction,dt::Float64,T::Float64,barostat::Barostat,gamma0::Float64,gammav::Float64,n::Int=4)
        fcell.ifrmat=false
        fcell.iffmat=false
        kb=8.617332385e-5 #eV/K
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
        mi=fcell.atoms[1].mass
        fcell.Volume=Vdt
        barostat.V=Vdt
        ltv=fcell.lattice_vectors
        for i in 1:natom
            pl[1,i]=fcell.atoms[i].momentum
            rl[1,i]=ltv*fcell.atoms[i].position
        end
        fcell.lattice_vectors=fcell.lattice_vectors.*(Vdt/V0)^(1/3)
        for j in 1:n-1
            fl=fill(zeros(3),length(fcell.atoms))
            for i in 1:length(fcell.atoms)
                fi=cell_forcei(fcell,interaction,i)
                fl[i]=fi
            end
            for i in 1:length(fcell.atoms)
                # if j==1
                #     pl[2*n+1,i]=pl[1,i].+(dt/2)*fl[i].+sqrt(kb*T*gamma0*dt)*rand(dist,3)-(gamma0*dt/2/mi)*pl[1,i]
                # end
                pl[2*j,i]=pl[2*j-1,i]+(ddt/2).*fl[i]+sqrt(kb*T*gamma0*ddt)*rand(dist,3)-(gamma0*ddt/2/mi)*pl[j,i]
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

end

