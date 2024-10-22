"""
Cu的EAM经验势,采用Ref中的Cu1势
Ref:Mendelev, M. I., Kramer, M. J., Becker, C. A., & Asta, M. (2008). Analysis of semi-empirical interatomic potentials appropriate for simulation of crystalline and liquid Al and Cu. Philosophical Magazine, 88(12), 1723–1750. https://doi.org/10.1080/14786430802206482
"""
module EMACu
using ..Model
using LinearAlgebra

export EMACu_phi,EMACu_psi,EMACu_Phi,EMACu_phi_gradient,EMACu_psi_gradient,EMACu_Phi_gradient,EMACu_rhoi,embedding_energyCu,embedding_forceCuij,embedding_forceCui

# EMACu_phi(r) 函数
function EMACu_phi(r)
    phi = 0.0
    if r <= 1.8
        phi += exp(11.026565103477 - 10.167211017722 * r + 6.0017702915006 * r^2 - 1.9598299733506 * r^3)
    end
    if 1.8 < r <= 2.8
        phi += 3.3519281301971 * (2.8 - r)^4 - 47.447602323833 * (2.8 - r)^5 + 111.06454537813 * (2.8 - r)^6 - 122.56379390195 * (2.8 - r)^7 + 49.14572206502 * (2.8 - r)^8
    end
    if 1.8 < r <= 4.8
        phi += 4.0605833179061 * (4.8 - r)^4 + 2.5958091214976 * (4.8 - r)^5 + 5.5656604545299 * (4.8 - r)^6 + 1.5184323060743 * (4.8 - r)^7 + 0.39696001635415 * (4.8 - r)^8
    end
    if 1.8 < r <= 6.0
        phi += -0.21402913758299 * (6.0 - r)^4 + 1.1714811538458 * (6.0 - r)^5 - 1.9913969426765 * (6.0 - r)^6 + 1.3862043035438 * (6.0 - r)^7 - 0.34520315264743 * (6.0 - r)^8
    end
    return phi
end

# EMACu_psi(r) 函数
function EMACu_psi(r)
    psi = 0.0
    if 0 <= r <= 2.4
        psi += 0.0199999875362 * (2.4 - r)^4
    end
    if 0 <= r <= 3.2
        psi += 0.019987533420669 * (3.2 - r)^4
    end
    if 0 <= r <= 4.5
        psi += 0.018861676713565 * (4.5 - r)^4
    end
    if 0 <= r <= 6.0
        psi += 0.0066082982694659 * (6.0 - r)^4
    end
    return psi
end

# EMACu_Phi(rho) 函数
function EMACu_Phi(rho)
    Phi = 0.0
    if rho >= 0
        Phi += -rho^0.5
    end
    if rho>=9.0 
        Phi += -5.7112865649408e-5 * (rho - 9)^4
    end
    if 11 <= rho 
        Phi += 3.0303487333648e-4 * (rho - 11)^4
    end
    if 13 <= rho 
        Phi += -5.4720795296134e-4 * (rho - 13)^4
    end
    if 15 <= rho
        Phi += 4.6278681464721e-4 * (rho - 15)^4
    end
    if 16 <= rho
        Phi += -1.0310712451906e-4 * (rho - 16)^4
    end
    if 16.5 <= rho
        Phi += 3.0634000239833e-3 * (rho - 16.5)^4
    end
    if 17 <= rho 
        Phi += -2.8308102136994e-3 * (rho - 17)^4
    end
    if 18 <= rho
        Phi += 6.4044567482688e-4 * (rho - 18)^4
    end
    return Phi
end




# EMACu_phi(r) 的梯度函数
function EMACu_phi_gradient(vec_r::Vector{Float64})
    dphi_dr = 0.0
    r=norm(vec_r)
    if 1 <= r <= 1.8
        exp_term = exp(11.026565103477 - 10.167211017722 * r + 6.0017702915006 * r^2 - 1.9598299733506 * r^3)
        dphi_dr += exp_term * (-10.167211017722 + 2 * 6.0017702915006 * r - 3 * 1.9598299733506 * r^2)
    end
    if 1.8 < r <= 2.8
        dphi_dr += -4 * 3.3519281301971 * (2.8 - r)^3 + 5 * 47.447602323833 * (2.8 - r)^4 - 6 * 111.06454537813 * (2.8 - r)^5 + 7 * 122.56379390195 * (2.8 - r)^6 - 8 * 49.14572206502 * (2.8 - r)^7
    end
    if 1.8 < r <= 4.8
        dphi_dr += -4 * 4.0605833179061 * (4.8 - r)^3 - 5 * 2.5958091214976 * (4.8 - r)^4 - 6 * 5.5656604545299 * (4.8 - r)^5 - 7 * 1.5184323060743 * (4.8 - r)^6 - 8 * 0.39696001635415 * (4.8 - r)^7
    end
    if 1.8 < r <= 6.0
        dphi_dr += 4 * 0.21402913758299 * (6.0 - r)^3 - 5 * 1.1714811538458 * (6.0 - r)^4 + 6 * 1.9913969426765 * (6.0 - r)^5 - 7 * 1.3862043035438 * (6.0 - r)^6 + 8 * 0.34520315264743 * (6.0 - r)^7
    end
    return dphi_dr * vec_r / r
end

# EMACu_psi(r) 的梯度函数
function EMACu_psi_gradient(vec_r::Vector{Float64})
    dpsi_dr = 0.0
    r=norm(vec_r)
    if 0 <= r <= 2.4
        dpsi_dr += -4 * 0.0199999875362 * (2.4 - r)^3
    end
    if 0 <= r <= 3.2
        dpsi_dr += -4 * 0.019987533420669 * (3.2 - r)^3
    end
    if 0 <= r <= 4.5
        dpsi_dr += -4 * 0.018861676713565 * (4.5 - r)^3
    end
    if 0 <= r <= 6.0
        dpsi_dr += -4 * 0.0066082982694659 * (6.0 - r)^3
    end
    return dpsi_dr * vec_r / r
end



# EMACu_Phi(rho) 的梯度函数
function EMACu_Phi_gradient(rho)
    dPhi_drho = 0.0
    if 0 <= rho 
        dPhi_drho += -0.5 * rho^(-0.5)
    end
    if 9 <= rho 
        dPhi_drho += -4 * 5.7112865649408e-5 * (rho - 9)^3
    end
    if 11 <= rho 
        dPhi_drho += 4 * 3.0303487333648e-4 * (rho - 11)^3
    end
    if 13 <= rho 
        dPhi_drho += -4 * 5.4720795296134e-4 * (rho - 13)^3
    end
    if 15 <= rho 
        dPhi_drho += 4 * 4.6278681464721e-4 * (rho - 15)^3
    end
    if 16 <= rho 
        dPhi_drho += -4 * 1.0310712451906e-4 * (rho - 16)^3
    end
    if 16.5 <= rho 
        dPhi_drho += 4 * 3.0634000239833e-3 * (rho - 16.5)^3
    end
    if 17 <= rho
        dPhi_drho += -4 * 2.8308102136994e-3 * (rho - 17)^3
    end
    if 18 <= rho
        dPhi_drho += 4 * 6.4044567482688e-4 * (rho - 18)^3
    end
    return dPhi_drho 
end


function EMACu_rhoi(cell::UnitCell,i;ct::Float64=6.0)
    natom=length(cell.atoms)
    rhoi=0.0
    for j in 1:natom
        if j!=i
            rij=getrij(cell,i,j)
            nrij=norm(rij)
            if nrij<ct
                # println("$i,$j,$rij,$(EMACu_psi(nrij))")
                rhoi+=EMACu_psi(nrij)
            end
        end
    end
    return rhoi
end


function embedding_energyCu(cell::UnitCell)
    natom=length(cell.atoms)
    E=0.0
    for i in 1:natom
        rhoi=EMACu_rhoi(cell,i)
        E+=EMACu_Phi(rhoi)
    end
    return E
end


function embedding_forceCuij(cell::UnitCell,i::Int,j::Int;ct::Float64=6.0)
    rij=getrij(cell,i,j)
    Fij=zeros(3)
    if norm(rij)<ct
        rhoi=EMACu_rhoi(cell,i)
        rhoj=EMACu_rhoi(cell,j)
        Fij=-(EMACu_Phi_gradient(rhoi)+EMACu_Phi_gradient(rhoj))*EMACu_psi_gradient(rij)
    end
    return Fij
end

function embedding_forceCui(cell,i::Int)
    natom=length(cell.atoms)

    Fi=zeros(3)
    for j in 1:natom
        if j!=i
            Fi+=embedding_forceCuij(cell,i,j)
        end
    end
    return Fi 
end

end