
"""
该势能计算过于复杂,暂时放弃,请使用EMA

Ref:Lee, B.-J., Shim, J.-H., & Baskes, M. I. (2003). Semiempirical atomic potentials for the fcc metals Cu, Ag, Au, Ni, Pd, Pt, Al, and Pb based on first and second nearest-neighbor modified embedded atom method. Physical Review B, 68(14), 144112. https://doi.org/10.1103/PhysRevB.68.144112
"""
module MEMA

using StaticArrays

using LinearAlgebra

using GLMakie 
using LsqFit
using ..Model
using ..MD
using DelimitedFiles
using Distributions
using Statistics
using Plots

export rho0,rho1sq,rho2sq,rho3sq,gammai,rhoi,drhoa,drho0,drho1sq,drho2sq,drho3sq,Ngradiant,dgammai,drhoi,Frho,dFrho

##default Cu:
beta0=3.83
beta1=2.2
beta2=6.0
beta3=2.2
t1=2.72
t2=3.04
t3=1.95 
B=1.420 #eV
A=0.94
Ec=3.54
Re=2.555

function  rhoa(R::Float64,beta::Float64,Re::Float64)
    return exp(-beta*(R/Re-1))
end

function rho0(cell::UnitCell,i::Int,beta0::Float64=beta0,Re::Float64=Re;rc::Float64=10.0)
    natom=length(cell.atoms)
    ltv=cell.lattice_vectors
    re=0.0
    for j=1:natom
        if i!=j
            nrij=norm(ltv*(cell.atoms[j].position-cell.atoms[i].position))
            if nrij<rc
                re+=rhoa(nrij,beta0,Re)
                # println("nrij=$nrij,i:$i,j:$j")
            end
        end
    end
    return re
    end


function rho1sq(cell::UnitCell,i::Int,beta1::Float64=beta1,Re::Float64=Re;rc::Float64=10.0)
    natom=length(cell.atoms)
    ltv=cell.lattice_vectors
    ree=0.0
    for a in 1:3
    re=0.0
        for j in 1:natom
        if i!=j
            rij=(ltv*(cell.atoms[j].position-cell.atoms[i].position))
            nrij=norm(rij)
            if nrij<rc
                re+=rhoa(norm(rij),beta1,Re)*rij[a]/nrij
            end
        end
        end
    ree+=re^2
    end
    return ree
    end


function rho2sq(cell::UnitCell,i::Int,beta2::Float64=beta2,Re::Float64=Re;rc::Float64=10.0)
natom=length(cell.atoms)
ltv=cell.lattice_vectors
ree=0.0
re2=0.0
    for a in 1:3
        for b in 1:3
            re=0.0
            for j in 1:natom
                if i!=j
                    rij=(ltv*(cell.atoms[j].position-cell.atoms[i].position))
                    nrij=norm(rij)
                    if nrij<rc
                        re+=rhoa(nrij,beta2,Re)*rij[a]*rij[b]/nrij^2
                    end

                    # if a==1&&b==1&&(nrij<rc)
                    #     re2+=rhoa(nrij,beta2,Re)
                    # end

                end
            end
            ree+=re^2
        end
    end
    return ree-(re2^2)/3
end

function rho3sq(cell::UnitCell,i::Int,beta3::Float64=beta3,Re::Float64=Re;rc::Float64=10.0)

natom=length(cell.atoms)
ltv=cell.lattice_vectors
reaa=0.0
re=0.0
for c=1:3
    for b=1:3
        for a=1:3
            ree=0.0
            rea=0.0
            for j in 1:natom
                if i!=j
                    rij=(ltv*(cell.atoms[j].position-cell.atoms[i].position))
                    nrij=norm(rij)
                    if nrij<rc
                        ree+=rhoa(nrij,beta3,Re)*rij[a]*rij[b]*rij[c]/nrij^3

                        if b==1&&c==1&&(nrij<rc)
                            rea+=rhoa(nrij,beta3,Re)*rij[a]/nrij
                        end
                    end
                end
            end
            if b==1&&c==1
                reaa+=rea^2
            end
            re+=ree^2
        end
    end

end
return re-reaa*0.6
end


function gammai(cell::UnitCell,i::Int,t::Vector{Float64}=[t1,t2,t3],beta::Vector{Float64}=[beta0,beta1,beta2,beta3],Re::Float64=Re;rc::Float64=10.0)
    g=0.0
  
    g=t[1]*(rho1sq(cell,i,beta[2],Re,rc=rc))+t[2]*(rho2sq(cell,i,beta[3],Re,rc=rc))+t[3]*(rho3sq(cell,i,beta[4],Re,rc=rc))
 
   
    return g/rho0(cell,i,beta[1],Re,rc=rc)^2
end


function rhoi(cell::UnitCell,i::Int,t::Vector{Float64}=[t1,t2,t3],beta::Vector{Float64}=[beta0,beta1,beta2,beta3],Re::Float64=Re;rc::Float64=10.0)
    return rho0(cell,i,beta[1],Re,rc=rc)*2/(1+exp(-gammai(cell,i,t,beta,Re,rc=rc))) 
end


function drhoa(R::Vector{Float64},beta::Float64,Re::Float64)
    nR=norm(R)
    return (rhoa(nR,beta,Re)/nR*beta/Re)*R
end


function drho0(cell::UnitCell,i::Int,beta0::Float64=beta0,Re::Float64=Re; rc::Float64=10.0)
    natom=length(cell.atoms)
    ltv=cell.lattice_vectors
    re=zeros(3)
    for j=1:natom
        if i!=j
            rij=ltv*(cell.atoms[j].position-cell.atoms[i].position)
            nrij=norm(rij)
            if nrij<rc
                # re-=(beta0*rhoa(nrij,beta0,Re)/nrij/Re)*rij
                re+=drhoa(rij,beta0,Re)
            end
        end
    end
    return re
    end

function drho1sq(cell::UnitCell,i::Int,beta1::Float64=beta1,Re::Float64=Re;rc::Float64=10.0)
natom=length(cell.atoms)
ltv=cell.lattice_vectors
ree=zeros(3)
for a in 1:3
re1=zeros(3)
re2=0.0
    for j in 1:natom
    if i!=j
        rij=(ltv*(cell.atoms[j].position-cell.atoms[i].position))
        nrij=norm(rij)
        if nrij<rc
            rhoaj=rhoa(nrij,beta1,Re)
            drhoaj=drhoa(rij,beta1,Re)
            re1+=(rij[a]*(1/nrij^3*rhoaj*rij+drhoaj/nrij)-I(3)[a,:]*rhoaj/nrij)
            re2+=rhoaj*rij[a]/nrij
        end
    end
    end
ree+=2*re1*re2
end
return ree
end


function drho2sq(cell::UnitCell,i::Int,beta2::Float64=beta2,Re::Float64=Re;rc::Float64=10.0)
    natom=length(cell.atoms)
    ltv=cell.lattice_vectors
    ree1=zeros(3)
    re2=zeros(3)
    re20=0.0

    for a in 1:3
        for b in 1:3
            re1=zeros(3)
            re10=0.0
            
            for j in 1:natom
                if i!=j
                    rij=(ltv*(cell.atoms[j].position-cell.atoms[i].position))
                    nrij=norm(rij)
                    rhoaj=rhoa(nrij,beta2,Re)
                    drhoaj=drhoa(rij,beta2,Re)
                    if nrij<rc
                        re1+=(rij[a]*rij[b]*(2*rij/nrij^4*rhoaj+drhoaj/nrij^2))-(I(3)[a,:]*rij[b]+I(3)[b,:]*rij[a])*rhoaj/nrij^2
                        re10+=rhoaj*rij[a]*rij[b]/nrij^2

                    end

                    # if a==1&&b==1&&(nrij<rc)
                    #     re2+=drhoaj
                    #     re20+=rhoaj
                    # end

                end
            end
            ree1+=re10*re1
        end
    end
    return 2*(ree1-1/3*re20*re2)
end


function drho3sq(cell::UnitCell,i::Int,beta3::Float64=beta3,Re::Float64=Re;rc::Float64=10.0)

    natom=length(cell.atoms)
    ltv=cell.lattice_vectors

    ree1=zeros(3)
    ree2=zeros(3)
    for c=1:3
        for b=1:3
            for a=1:3
                re1=zeros(3)
                re10=0.0
                re2=zeros(3)
                re20=0.0
                for j in 1:natom
                    if i!=j
                        rij=(ltv*(cell.atoms[j].position-cell.atoms[i].position))
                        nrij=norm(rij)
                        if nrij<rc
                            rhoaj=rhoa(nrij,beta3,Re)
                            drhoaj=drhoa(rij,beta3,Re)
                            re1+=(rij[a]*rij[b]*rij[c]*(3*rij/nrij^5*rhoaj+drhoaj/nrij^3))-(I(3)[a,:]*rij[b]*rij[c]+I(3)[b,:]*rij[a]*rij[c]+I(3)[c,:]*rij[a]*rij[b])*rhoaj/nrij^3
                            re10+=rhoaj*rij[a]*rij[b]*rij[c]/nrij^3

                            if b==1&&c==1
                                re2+=rij[a]*(drhoaj/nrij+rhoaj*rij/nrij^3)-I(3)[a,:]*rhoaj/nrij
                                re20+=rhoaj*rij[a]/nrij
                                
                            end
                        end
                    end
                end
                if b==1&&c==1
                    ree2+=2*re2*re20
                end
                ree1+=2*re10*re1
            end
        end
    
    end
    return ree1-ree2*0.6
    end
function  Ngradiant(cell::UnitCell,i::Int,f::Function,para::Vector;dr::Vector{Float64}=[0.001,0.001,0.001])
    df=zeros(3)
    lt=cell.lattice_vectors
    invlt=inv(lt)
    drm=Diagonal(dr)
    for j in 1:3
        dri=invlt*drm[j,:]
        dcell=deepcopy(cell)
        dcell.atoms[i].position+=dri
        f1=f(dcell,i,para...)
        dcell.atoms[i].position-=2*dri
        f2=f(dcell,i,para...)
        df[j]=(f1-f2)/2/dr[j]
    end

    return df
end

function dgammai(cell::UnitCell,i::Int,t::Vector{Float64}=[t1,t2,t3],beta::Vector{Float64}=[beta0,beta1,beta2,beta3],Re::Float64=Re;rc::Float64=10.0)
    dg=zeros(3)
  
    rho1sqi=rho1sq(cell,i,beta[2],Re,rc=rc)
    rho2sqi=rho2sq(cell,i,beta[3],Re,rc=rc)
    rho3sqi=rho3sq(cell,i,beta[4],Re,rc=rc)
    rho0i=rho0(cell,i,beta[1],Re,rc=rc)
    drho1sqi=drho1sq(cell,i,beta[2],Re,rc=rc)
    drho2sqi=drho2sq(cell,i,beta[3],Re,rc=rc)
    drho3sqi=drho3sq(cell,i,beta[4],Re,rc=rc)
    drho0i=drho0(cell,i,beta[1],Re,rc=rc)
    dg+=t[1]*(drho1sqi*(rho0i)-2*rho1sqi*drho0i)/rho0i^3
    dg+=t[2]*(drho2sqi*(rho0i)-2*rho2sqi*drho0i)/rho0i^3
    dg+=t[3]*(drho3sqi*(rho0i)-2*rho3sqi*drho0i)/rho0i^3
    return dg
end

function drhoi(cell::UnitCell,i::Int,t::Vector{Float64}=[t1,t2,t3],beta::Vector{Float64}=[beta0,beta1,beta2,beta3],Re::Float64=Re;rc::Float64=10.0)
    rho0i=rho0(cell,i,beta[1],Re,rc=rc)
    gi=gammai(cell,i,t,beta,Re,rc=rc)
    dg=dgammai(cell,i,t,beta,Re,rc=rc)
    drho0i=drho0(cell,i,beta[1],Re,rc=rc)
    dr=2*drho0i/(1+exp(-gi))+2*rho0i/(1+exp(-gi))^2*exp(-gi)*dg
    return dr
end

function Frho(cell::UnitCell,i::Int,t::Vector{Float64}=[t1,t2,t3],beta::Vector{Float64}=[beta0,beta1,beta2,beta3],Re::Float64=Re,Ec::Float64=Ec,A::Float64=A,r0::Float64=1.0;rc::Float64=10.0)
    k=rhoi(cell,i,t,beta,Re,rc=rc)/r0
    return A*Ec*(k)*log(k)
end
function dFrho(cell::UnitCell,i::Int,t::Vector{Float64}=[t1,t2,t3],beta::Vector{Float64}=[beta0,beta1,beta2,beta3],Re::Float64=Re,Ec::Float64=Ec,A::Float64=A,r0::Float64=1.0;rc::Float64=10.0)

    dri=drhoi(cell,i,t,beta,Re,rc=rc)
    ri=rhoi(cell,i,t,beta,Re,rc=rc)
    k=ri/r0
 dF=A*Ec*(1+log(k))*(dri/r0)
    return dF
end

end