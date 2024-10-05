module MD

using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
include("modle.jl")
using .Model
    


function dUdV_default(r::Vector{Float64},v::Float64)
    return 0.0
end

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


struct Thermostat
    T::Float64  # 目标温度
    Q::Float64  # 热浴质量
    Rt::Float64  # 热浴变量 (friction coefficient)
    Pt::Float64  # 热浴动量
end
struct Barostat
    P::Float64  # 目标压力
    V::Float64  # 系统体积
    Q::Float64  # 压力浴质量
    Rp::Float64  # 压力浴变量 (friction coefficient)
    Pp::Float64  # 压力浴动量
end

##保留原始版本hz
# function Hz(z::Vector{Float64},cell::UnitCell,interaction::Interaction,thermostat::Thermostat,barostat::Barostat,dUdV::T=dUdV_default) where T
#     global kb
#     dim=3*length(cell.atoms)+2
#     Hz=zeros(dim*2)
#     natom=length(cell.atoms)
#     v=cell.Volume
#     W=barostat.W
#     Q=thermostat.Q
#     temp=thermostat.T
#     Pe=barostat.Pe
#     Pint=pressure_int(cell,interaction,dUdV)
#     addp=sum([dot(atom.momentum,atom.momentum)/atom.mass for atom in cell.atoms])
#     for i in 1:natom
#         atom=cell.atoms[i]
#         mi=atom.mass
#         Hz[3*i+1]=1/mi+z[2*dim]*z[3*i+1]/W
#         Hz[3*i+2]=1/mi+z[2*dim]*z[3*i+2]/W
#         Hz[3*i+3]=1/mi+z[2*dim]*z[3*i+3]/W
#         fi=cell_forcei(cell,interaction,i)
#         Hz[dim+3*i-2]=fi[1]-(1+1/natom)*z[2*dim]/W*z[dim+3*i-2]-z[2*dim-1]*z[2*dim]/Q
#         Hz[dim]=3*z[dim]*z[2*dim]/W
#         Hz[2*dim]=3*z[dim]*(Pint-Pe)+1/natom*addp-z[2*dim-1]*z[2*dim]/Q
#         z[dim-1]=z[2*dim-1]/Q
#         z[2*dim-1]=addp+z[2*dim]^2/W-(3*natom+1)*kb*temp
#     end

#     return Hz
# end

end