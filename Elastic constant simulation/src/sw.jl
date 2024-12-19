"""
SW potential for diamond are implemented in this module.
*the force are not calculated well yet,only SWforcei_Diamond can give correct force by nurmerical differentiation.
Ref:Barnard, A. S., & Russo, S. P. (2002). Development of an improved Stillinger-Weber potential for tetrahedral carbon using ab initio (Hartree-Fock and MP2) methods. Molecular Physics, 100(10), 1517–1525. https://doi.org/10.1080/00268970110109853
"""
module SWPotential
using StaticArrays
using LinearAlgebra
using ..Model
export v2Diamond,v2gradient_Diamond,v3Diamond,SWenergy_Diamodijk,SWenergy_Diamond,SWforce_Diamond,hDiamond,dhDiamonddtheta,dhDiamonddrij,dhDiamonddrik,fMatrix,SWforce_Diamondijk0,SWforcei_Diamond

A=5.3789794
B=0.5933864
p=4.0
q=0.0
a=1.846285
lambda=26.19934
gamma=1.055116
sigma=1.368 #A 
# epsilon=21682051.15 #eV
epsilon=3.551 #eV
ct=a*sigma

function v2Diamond(r::Float64)
    x=r/sigma
    if x<a
        return epsilon*A*(B/x^p-1)*exp(1/(x-a))
    else
        return 0.0
    end
end

function v2gradient_Diamond(vec_r::SVector{3, Float64})
    r=norm(vec_r)
    x=r/sigma
    if x<a
    dv2=epsilon*A*(-p*B/x^(p+1)-1/(x-a)^2*(B/x^p-1))*exp(1/(x-a))
    return -dv2/sigma*vec_r/r
    else
        return SVector{3,Float64}(0.0,0.0,0.0)
    end
end



function hDiamond(rij::SVector{3,Float64},rik::SVector{3,Float64})
    nrij=norm(rij)
    nrik=norm(rik)
    as=a*sigma
    if nrij>as || nrik>as
        return 0.0
    end
    cos_theta = dot(rij, rik) / nrij/nrik
    return epsilon*lambda*exp(gamma*sigma/(nrij-as)+gamma*sigma/(nrik-as))*(cos_theta+1/3)^2
end


# function hDiamond(rij::SVector{3,Float64},rik::SVector{3,Float64})
#     rij0=rij./sigma
#     rik0=rik./sigma
#     nrij0=norm(rij0)
#     nrik0=norm(rik0)
#     if nrij0>a || nrik0>a
#         return 0.0
#     end
#     cos_theta = dot(rij0, rik0) / (nrij0 * nrik0)
#     return lambda*exp(gamma/(nrij0-a)+gamma/(nrik0-a))*(cos_theta+1/3)^2
# end

function v3Diamond(rij::SVector{3,Float64},rik::SVector{3,Float64})
    rjk=rik-rij
    return (hDiamond(rij,rik)+hDiamond(-rij,rjk)+hDiamond(-rik,-rjk))
end


# function SWenergy_Diamodijk(cell::UnitCell,i::Int64,j::Int64,k::Int64)
#     if i!=j && i!=k && j!=k
#         rij=getrij(cell,i,j)
#         rik=getrij(cell,i,k)
#         energy=v3Diamond(rij,rik)
#     else
#         energy=0.0
#     end
#     return energy
# end

# function SWenergy_Diamond(cell::UnitCell)
#     energy=0.0
#     natom=length(cell.atoms)
#     for i in 1:natom
#         for j in i+1:natom
#             if i!=j
#             for k in j+1:natom
#                 energy+=SWenergy_Diamodijk(cell,i,j,k)
#             end
#         end
#         end
#     end
#     return energy
# end

function SWenergy_Diamodijk(cell::UnitCell,i::Int64,j::Int64,k::Int64)
    if i!=j && i!=k && j!=k
        rij=getrij(cell,i,j)
        rik=getrij(cell,i,k)
        energy=hDiamond(rij,rik)
    else
        energy=0.0
    end
    return energy
end

function SWenergy_Diamond(cell::UnitCell)
    energy=0.0
    natom=length(cell.atoms)
    for i in 1:natom
        for j in i+1:natom
            for k in j+1:natom
                    if i!=j!=k
                        energy+=SWenergy_Diamodijk(cell,i,j,k)
                        energy+=SWenergy_Diamodijk(cell,j,k,i)
                        energy+=SWenergy_Diamodijk(cell,k,i,j)
                    end
                end
            end
    end
    return energy
end

function SWforcei_Diamond(cell::UnitCell,i::Int)
    dr=[0.001,0.001,0.001]
    df=zeros(3)
    lt=cell.lattice_vectors
    invlt=inv(lt)
    drm=Diagonal(dr)
    for j in 1:3
        dri=invlt*drm[j,:]
        dcell=deepcopy(cell)
        dcell.atoms[i].position.+=dri
        update_rmati!(dcell,i)
        f1=SWenergy_Diamond(dcell)
        # println(f1)
        dcell.atoms[i].position.-=2*dri
        update_rmati!(dcell,i)
        f2=SWenergy_Diamond(dcell)
        df[j]=(f1-f2)/2/dr[j]
    end
    return -SVector{3,Float64}(df)
end

function dhDiamonddtheta(rij::SVector{3,Float64},rik::SVector{3,Float64})
    nrij=norm(rij)
    nrik=norm(rik)
    as=a*sigma
    if nrij>as || nrik>as
        return 0.0
    end
    cos_theta = dot(rij, rik) / nrij/nrik
    return epsilon*lambda*exp(gamma*sigma/(nrij-as)+gamma*sigma/(nrik-as))*(cos_theta+1/3)*(-2)
end

function dhDiamonddrij(rij::SVector{3,Float64},rik::SVector{3,Float64})
    nrij=norm(rij)
    nrik=norm(rik)
    as=a*sigma
    if nrij>as || nrik>as
        return 0.0
    end
    cos_theta = dot(rij, rik) / nrij/nrik
    return epsilon*lambda*exp(gamma*sigma/(nrij-as)+gamma*sigma/(nrik-as))*(cos_theta+1/3)^2*(-gamma*sigma/(nrij-as)^2)
end

function dhDiamonddrik(rij::SVector{3,Float64},rik::SVector{3,Float64})
    nrij=norm(rij)
    nrik=norm(rik)
    as=a*sigma
    if nrij>as || nrik>as
        return 0.0
    end
    cos_theta = dot(rij, rik) / nrij/nrik
    return epsilon*lambda*exp(gamma*sigma/(nrij-as)+gamma*sigma/(nrik-as))*(cos_theta+1/3)^2*(-gamma*sigma/(nrik-as)^2)
end

function fMatrix(rij::SVector{3,Float64},rik::SVector{3,Float64})
    fmat=zeros(3,3)
    nrij=norm(rij)
    nrik=norm(rik)
    costheta=dot(rij,rik)/(nrij*nrik)
    fi1=1/nrij*(dhDiamonddrij(rij,rik))-(1/nrij/nrik-costheta/nrij^2)*(dhDiamonddtheta(rij,rik))
    fi2=1/nrik*(dhDiamonddrik(rij,rik))-epsilon*(1/nrik/nrij-costheta/nrik^2)*(dhDiamonddtheta(rij,rik))
    fj2=(dhDiamonddtheta(rij,rik))/nrij/nrik
    fmat[1,1]=fi1
    fmat[1,2]=fi2
    fmat[2,1]=-fi1
    fmat[2,3]=fj2
    fmat[3,2]=-fi2
    fmat[3,3]=-fj2
    return fmat
end

function SWforce_Diamondijk0(cell::UnitCell,i::Int64,j::Int64,k::Int64)
    if i!=j && i!=k && j!=k
        rij=getrij(cell,i,j)
        rik=getrij(cell,i,k)
        rjk=rij-rik
        fmat=fMatrix(rij,rik)
        fi=fmat[1,1]*rij+fmat[1,2]*rik
        fj=fmat[2,1]*rij+fmat[2,3]*rik
        fk=fmat[3,2]*rik+fmat[3,3]*rjk
        # println("i=",i,"j=",j,"k=",k,"fi=",fi,"fj=",fj,"fk=",fk)
    else
        return (zeros(3),zeros(3),zeros(3))
    end
    return (fi,fj,fk)
end


function SWforce_Diamond(cell::UnitCell)
    natom=length(cell.atoms)
    cfmat = fill(SVector{3, Float64}(0.0, 0.0, 0.0), natom)
    for i in 1:natom
        for j in i+1:natom
            for k in j+1:natom
                    if i!=j!=k
                            fi,fj,fk=SWforce_Diamondijk0(cell,i,j,k)
                            cfmat[i]+=fi
                            cfmat[j]+=fj
                            cfmat[k]+=fk
                            fi,fj,fk=SWforce_Diamondijk0(cell,j,k,i)
                            cfmat[j]+=fi
                            cfmat[k]+=fj
                            cfmat[i]+=fk
                            fi,fj,fk=SWforce_Diamondijk0(cell,k,i,j)
                            cfmat[k]+=fi
                            cfmat[i]+=fj
                            cfmat[j]+=fk
                        end
                    end
            end
    end
    return cfmat

end

end