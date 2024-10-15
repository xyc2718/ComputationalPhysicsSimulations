module MD
using Distributions
using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using ..Model
using Base.Threads
    

export pressure_int,Thermostat,Barostat,Hz,symplectic_matrix,RK3_step!,z2atoms,z2cell,cell2z,dUdV_default,update_cell! ,zmod!,initT!,initcell
###MD


"""
default function for dUdV
:param r: position of atom
:param v: volume of cell
return 0.0
"""
function dUdV_default(inicell::UnitCell,interaction::Interaction,dV::BigFloat=BigFloat("1e-9"))
    natom=length(inicell.atoms)
    V0::BigFloat=inicell.Volume
    dV0::BigFloat=inicell.Volume*dV
    V1=V0+dV0
    V2=V0-dV0
    ltv=inicell.lattice_vectors
    ltv1=ltv*(V1/V0)^(1/3)
    ltv2=ltv*(V2/V0)^(1/3)

    dcell=deepcopy(inicell)
    dcell.lattice_vectors=ltv1
    for i in natom
    dcell.atoms[i].position=inv(ltv1)*ltv*dcell.atoms[i].position
    end
    energy1=cell_energy0(dcell,interaction)

    dcell=deepcopy(inicell)
    dcell.lattice_vectors=ltv2
    for i in natom
    dcell.atoms[i].position=inv(ltv2)*ltv*dcell.atoms[i].position
    end
    energy2=cell_energy0(dcell,interaction)


  return (energy1-energy2)/dV0/2
end


"""
等效压强Pint
:param cell: UnitCell
:param interaction: Interaction
:param dUdV: function for dUdV and default is dUdV_default return 0.0
return Pint
"""
function pressure_int(cell::UnitCell,interaction::Interaction,dUdV::T=dUdV_default) where T
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
恒温器
:param T: 目标温度
:param Q: 热浴质量
:param Rt: 热浴变量 (friction coefficient)
:param Pt: 热浴动量
"""
mutable struct Thermostat
    T::Float64  # 目标温度
    Q::Float64  # 热浴质量
    Rt::Float64  # 热浴变量 (friction coefficient)
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
    Pv::Float64  # 压力浴动量,logv
end


"""
计算dH/dz
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param cell: UnitCell
:param interaction: Interaction
:param thermostat: Thermostat
:param barostat: Barostat
:param dUdV: function for dUdV and default is dUdV_default return 0.0
"""
function Hz(z::Vector{Float64},cell::UnitCell,interaction::Interaction,thermostat::Thermostat,barostat::Barostat;dUdV::T=dUdV_default,kb::Float64=1.0) where T
    
    dim=3*length(cell.atoms)+2
    Hz=zeros(dim*2)
    natom=length(cell.atoms)
    # if dim!=length(z)
    #     throw("The dimension of z is not consist with the dimension of the system. z should be natom*3+2")
    # end
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
    # println()
    # println(3*z[dim]*(Pint-Pe))
    # println(1/natom*addp)
    # println(-z[2*dim-1]*z[2*dim]/Q)
    # println()

    Hz[dim-1]=z[2*dim-1]/Q
    Hz[2*dim-1]=addp+z[2*dim]^2/W-(3*natom+1)*kb*temp
    return Hz
end

"""
生成2n x 2n 的辛矩阵
"""
function symplectic_matrix(n::Int)
    I = Matrix{Float64}(I, n, n)  # 创建 n x n 的单位矩阵
    O = zeros(Float64, n, n)      # 创建 n x n 的零矩阵
    J = [O I; -I O]               # 使用块矩阵构建辛矩阵
    return J
end


"""
RK3步进
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param dt: 步长
:param cell: UnitCell
:param interaction: Interaction
:param thermostat: Thermostat
:param barostat: Barostat
:param dUdV: function for dUdV and default is dUdV_default return 0.0
return z
"""
function RK3_step!(z::Vector{Float64},dt::Float64,cell::UnitCell, interaction::Interaction, thermostat::Thermostat, barostat::Barostat;kb::Float64=1.0)
    k1=Hz(z,cell,interaction,thermostat,barostat)
    zr=z+(dt/2).*k1
    update_cell!(zr,cell)
    k2=Hz(zr,cell,interaction,thermostat,barostat)
    zr=z-dt.*k1+(2*dt).*k2
    update_cell!(zr,cell)
    k3=Hz(zr,cell,interaction,thermostat,barostat)
    z.=z+(dt/6).*(k1.+4*k2.+k3)
    update_cell!(z,cell)
    dim=Int(length(z)/2)
    thermostat.Pt=z[2*dim-1]
    barostat.Pv=z[2*dim]
    thermostat.Rt=z[dim-1]
    barostat.V=z[dim]
    end

function zmod!(z::Vector{Float64},cell::UnitCell)
    natom=length(cell.atoms)
    a,b,c=cell.lattice_vectors*cell.copy
    for i in 1:natom
        z[3*i-2]=mod(z[3*i-2],a)
        z[3*i-1]=mod(z[3*i-1],b)
        z[3*i]=mod(z[3*i],c)
        z[3*natom+3*i]=mod(z[3*natom+3*i],a)
        z[3*natom+3*i+1]=mod(z[3*natom+3*i+1],b)
        z[3*natom+3*i+2]=mod(z[3*natom+3*i+2],c)
    end
end
"""
将z转化为atoms
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
return atoms
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
"""
function z2cell(z::Vector{Float64},cell::UnitCell)
    atoms=z2atoms(z,cell)
    newcell=UnitCell(cell.lattice_vectors,atoms,cell.copy)
    return newcell
end


"""
将z转化为UnitCell,直接修改原始cell
:param z: z=[r1...rn,Rt,Rv,p1,...pn,Pt,Pv]
:param cell: UnitCell
"""
function update_cell!(z::Vector{Float64},cell::UnitCell)
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
    ltm=cell.lattice_vectors*ap  
    cell.lattice_vectors=ltm
    cell.Volume=v
    a,b,c=cell.copy
    # z[1:3*natom].=z[1:3*natom]
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





function initT!(T::Float64,cell::UnitCell) 
    m=cell.atoms[1].mass
    kb=1.0
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

    for i in 1:natom
        cell.atoms[i].momentum=cell.atoms[i].momentum.*sqrt(Ekstd/Ek)
    end
end


function initcell(P::Float64,T::Float64,atoms::Vector{Atom},interaction::Interaction;cp::Vector{Int}=[3,3,3],Prg::Vector{Float64}=[0.05,3.0])
    m=atoms[1].mass
    kb=1.0
    natom=length(atoms)
    a,b=Prg
    for i in 1:100
        lt=a
        lattice_vectors = (Matrix([
                            lt 0.0 0.0; #a1
                            0.0 lt 0.0; #a2
                            0.0 0.0 lt] #a3
                        ))
        cell=UnitCell(lattice_vectors,atoms)
        cpcell=copycell(cell,cp...)
        fcell=filtercell(cpcell)
        initT!(T,fcell)
        pa=pressure_int(fcell,interaction)-P
        
        lt=b
        lattice_vectors = (Matrix([
                            lt 0.0 0.0; #a1
                            0.0 lt 0.0; #a2
                            0.0 0.0 lt] #a3
                        ))
        cell=UnitCell(lattice_vectors,atoms)
        cpcell=copycell(cell,cp...)
        fcell=filtercell(cpcell)
        initT!(T,fcell)
        pb=pressure_int(fcell,interaction)-P
        if pa*pb>0
            throw("Pa,Pb符号相同 at a=$a,Pa=$pa,$b=b,Pb=$pb")
        end
        c=(a+b)/2
        lt=c
        lattice_vectors = (Matrix([
                            lt 0.0 0.0; #a1
                            0.0 lt 0.0; #a2
                            0.0 0.0 lt] #a3
                        ))
        cell=UnitCell(lattice_vectors,atoms)
        cpcell=copycell(cell,cp...)
        fcell=filtercell(cpcell)
        initT!(T,fcell)
        pc=pressure_int(fcell,interaction)-P
        if abs(pc)<1e-2
            return fcell
        end
        if pa*pc<0
            b=c
        else
            a=c
        end
    end
end
end

