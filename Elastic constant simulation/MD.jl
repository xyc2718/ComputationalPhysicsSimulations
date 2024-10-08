module MD

using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using ..Model
using Base.Threads
    

export pressure_int,Thermostat,Barostat,Hz,symplectic_matrix,RK3_step!,z2atoms,z2cell,cell2z,dUdV_default,update_cell!  
###MD


"""
default function for dUdV
:param r: position of atom
:param v: volume of cell
return 0.0
"""
function dUdV_default(r::Vector{Float64},v::Float64)
    return 0.0
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
    Pint=0.0
    for i in 1:length(cell.atoms)
        atom=cell.atoms[i]
        pm=atom.momentum
        ri=atom.position
        fi=cell_forcei(cell,interaction,i)
        Pint+=dot(pm,pm)/atom.mass+dot(ri,fi)-3*v*dUdV(atom.position,v)   
    end
    return Pint/v/3
    
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
    Pv::Float64  # 压力浴动量
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
function RK3_step!(z::Vector{Float64},dt::Float64,cell::UnitCell, interaction::Interaction, thermostat::Thermostat, barostat::Barostat;dUdV::T = dUdV_default,kb::Float64=1.0) where T
    k1=Hz(z,cell,interaction,thermostat,barostat,dUdV=dUdV,kb=kb)
    zr=z+(dt/2).*k1
    update_cell!(zr,cell)
    k2=Hz(zr,cell,interaction,thermostat,barostat,dUdV=dUdV,kb=kb)
    zr=z-dt.*k1+(2*dt).*k2
    update_cell!(zr,cell)
    k3=Hz(zr,cell,interaction,thermostat,barostat,dUdV=dUdV,kb=kb)
    z.=z+(dt/6).*(k1.+4*k2.+k3)
    update_cell!(z,cell)
    a,b,c=cell.lattice_vectors*cell.copy
    natom=length(cell.atoms)
    for i in 1:natom
        z[3*i-2]=mod(z[3*i-2],a)
        z[3*i-1]=mod(z[3*i-1],b)
        z[3*i]=mod(z[3*i],c)
        z[3*natom+3*i]=mod(z[3*natom+3*i],a)
        z[3*natom+3*i+1]=mod(z[3*natom+3*i+1],b)
        z[3*natom+3*i+2]=mod(z[3*natom+3*i+2],c)
    end
    z[3*natom+1]=z[3*natom+1]
    z[3*natom+2]=z[3*natom+2]
    z[2*3*natom+3]=z[2*3*natom+3]
    z[2*3*natom+4]=z[2*3*natom+4]
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
    ap=((v/v0)^(1/3)) 
    ltm=cell.lattice_vectors.*ap
    cell.lattice_vectors=ltm
    cell.Volume=v
#####################
    for i in 1:natom
        cell.atoms[i].position=inv(cell.lattice_vectors)*rl[3*i-2:3*i]
        cell.atoms[i].momentum=pl[3*i-2:3*i]
    end
    # z[1:3*natom]=z[1:3*natom].*ap
    # z[3*natom+3:3*natom+3*natom+3]=z[3*natom+3:3*natom+3*natom+3]
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
end

