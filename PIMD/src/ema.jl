"""
Cu和Al的EAM经验势,实现Ref中的Al1,Al2,Cu1势能,有embedding_energyX,embedding_forceiX,embedding_forceijX

Ref:Mendelev, M. I., Kramer, M. J., Becker, C. A., & Asta, M. (2008). Analysis of semi-empirical interatomic potentials appropriate for simulation of crystalline and liquid Al and Cu. Philosophical Magazine, 88(12), 1723–1750. https://doi.org/10.1080/14786430802206482
"""
module EMA
using ..Model
using LinearAlgebra
using StaticArrays
using Base.Threads
export EMACu_phi,EMACu_psi,EMACu_Phi,EMACu_phi_gradient,EMACu_psi_gradient,EMACu_Phi_gradient,EMACu_rhoi,embedding_energyCu,embedding_forceCuij,embedding_forceCui,EMAAl2_Phi,EMAAl2_phi,EMAAl2_psi,EMAAl2_phi_gradient,EMAAl2_psi_gradient,EMAAl2_Phi_gradient,EMAAl2_rhoi,embedding_energyAl2,embedding_forceAl2ij,embedding_forceAl2i,EMAAl1_Phi,EMAAl1_phi,EMAAl1_psi,EMAAl1_phi_gradient,EMAAl1_psi_gradient,EMAAl1_Phi_gradient,EMAAl1_rhoi,embedding_energyAl1,embedding_forceAl1ij,embedding_forceAl1i
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
function EMACu_phi_gradient(vec_r::SVector{3,Float64})
    dphi_dr = 0.0
    r=norm(vec_r)
    if  r <= 1.8
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
    return -dphi_dr * vec_r / r
end

# EMACu_psi(r) 的梯度函数
function EMACu_psi_gradient(vec_r::SVector{3,Float64})
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
    return -dpsi_dr * vec_r / r
end



# EMACu_Phi(rho) 的梯度函数
function EMACu_Phi_gradient(rho::Float64)
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


function embedding_forceCuij(cell::UnitCell,i::Int,j::Int;ct::Float64=6.0)::SVector{3,Float64}
    rij=getrij(cell,i,j)
    Fij=SVector{3,Float64}(0.0,0.0,0.0)
    if norm(rij)<ct
        rhoi=EMACu_rhoi(cell,i)
        rhoj=EMACu_rhoi(cell,j)
        Fij=-(EMACu_Phi_gradient(rhoi)+EMACu_Phi_gradient(rhoj))*EMACu_psi_gradient(rij)
    end
    return Fij
end

function embedding_forceCui(cell,i::Int)::SVector{3,Float64}
    natom=length(cell.atoms)

    Fi=SVector{3,Float64}(0.0,0.0,0.0)
    for j in 1:natom
        if j!=i
            Fi+=embedding_forceCuij(cell,i,j)
        end
    end
    return Fi 
end




function EMAAl2_phi(rij::Float64)::Float64
    phi = 0.0
    if rij ≤ 2.25
        phi += exp(1.0655898030717 + 6.9189333025554 * rij - 5.4560152009179 * rij^2 + 0.97305935423516 * rij^3)
    end
    
    if 2.25 < rij ≤ 3.2
        phi += 10.797831008871 * (3.2 - rij)^4 -
                38.354420072333 * (3.2 - rij)^5 +
                83.609733168086 * (3.2 - rij)^6 -
                75.644960845874 * (3.2 - rij)^7 +
                27.397628449176 * (3.2 - rij)^8
    end

    if 2.25 < rij ≤ 4.8
        phi += -1.6404275277304 * (4.8 - rij)^4 +
                1.9359384900534 * (4.8 - rij)^5 -
                2.3676607051992 * (4.8 - rij)^6 +
                0.68948838258734 * (4.8 - rij)^7 -
                0.14749445109681 * (4.8 - rij)^8
    end

    if 2.25 < rij ≤ 6.5
        phi += 0.19214771321964 * (6.5 - rij)^4 -
                0.40788777117632 * (6.5 - rij)^5 +
                0.33795215935241 * (6.5 - rij)^6 -
                0.12880925102229 * (6.5 - rij)^7 +
                0.019019373704492 * (6.5 - rij)^8
    end

    return phi
end

function EMAAl2_psi(rij::Float64)::Float64
    psi = 0.0
    if 0 ≤ rij ≤ 2.5
        psi += 0.00019850823042883 * (2.5 - rij)^4
    end
    if 0 ≤ rij ≤ 2.6
        psi += 0.10046665347629 * (2.6 - rij)^4
    end
    if 0 ≤ rij ≤ 2.7
        psi += 0.10054338881951 * (2.7 - rij)^4
    end
    if 0 ≤ rij ≤ 2.8
        psi += 0.099104582963213 * (2.8 - rij)^4
    end
    if 0 ≤ rij ≤ 3.0
        psi += 0.090086286376778 * (3.0 - rij)^4
    end
    if 0 ≤ rij ≤ 3.4
        psi += 0.0073022698419468 * (3.4 - rij)^4
    end
    if 0 ≤ rij ≤ 4.2
        psi += 0.014583614223199 * (4.2 - rij)^4
    end
    if 0 ≤ rij ≤ 4.8
        psi += -0.0010327381407070 * (4.8 - rij)^4
    end
    if 0 ≤ rij ≤ 5.6
        psi += 0.0073219994475288 * (5.6 - rij)^4
    end
    if 0 ≤ rij ≤ 6.5
        psi += 0.0095726042919017 * (6.5 - rij)^4
    end

    return psi
end

function EMAAl2_Phi(rho::Float64)::Float64
    Phi = -sqrt(rho)
    
    if rho >= 16
        Phi += 3.5025051308271e-4 * (rho - 16)^4 -
                6.9606881126760e-5 * (rho - 16)^5 +
                3.5717262505601e-6 * (rho - 16)^6
    end

    if rho >= 24
        Phi += -1.8828783364689e-3 * (rho - 24)^4 +
                1.1303817967915e-6 * (rho - 24)^5 -
                8.0158124278034e-6 * (rho - 24)^6
    end

    if rho >= 30
        Phi += 2.7235851330782e-4 * (rho - 30)^4 +
                4.8914043715808e-5 * (rho - 30)^5 +
                4.7226737586655e-6 * (rho - 30)^6
    end

    return Phi
end


function EMAAl2_phi_gradient(vrij::SVector{3,Float64})::SVector{3,Float64}
    dphi = 0.0
    rij = norm(vrij)
    
    if 1.6 ≤ rij ≤ 2.25
        dphi += (6.9189333025554 - 2 * 5.4560152009179 * rij + 3 * 0.97305935423516 * rij^2) *
            exp(1.0655898030717 + 6.9189333025554 * rij - 5.4560152009179 * rij^2 + 0.97305935423516 * rij^3)
    end

    if 2.25 < rij ≤ 3.2
        dphi += -4 * 10.797831008871 * (3.2 - rij)^3 + 
                5 * 38.354420072333 * (3.2 - rij)^4 -
                6 * 83.609733168086 * (3.2 - rij)^5 +
                7 * 75.644960845874 * (3.2 - rij)^6 -
                8 * 27.397628449176 * (3.2 - rij)^7
    end

    if 2.25 < rij ≤ 4.8
        dphi += 4 * 1.6404275277304 * (4.8 - rij)^3 -
                5 * 1.9359384900534 * (4.8 - rij)^4 +
                6 * 2.3676607051992 * (4.8 - rij)^5 -
                7 * 0.68948838258734 * (4.8 - rij)^6 +
                8 * 0.14749445109681 * (4.8 - rij)^7
    end

    if 2.25 < rij ≤ 6.5
        dphi += -4 * 0.19214771321964 * (6.5 - rij)^3 + 
                5 * 0.40788777117632 * (6.5 - rij)^4 -
                6 * 0.33795215935241 * (6.5 - rij)^5 +
                7 * 0.12880925102229 * (6.5 - rij)^6 -
                8 * 0.019019373704492 * (6.5 - rij)^7
    end

    return -dphi * vrij / rij
end

function EMAAl2_psi_gradient(vrij::SVector{3,Float64})::SVector{3,Float64}
    rij = norm(vrij)
    dpsi = 0.0
    
    if 0 ≤ rij ≤ 2.5
        dpsi += -4 * 0.00019850823042883 * (2.5 - rij)^3
    end
    if 0 ≤ rij ≤ 2.6
        dpsi += -4 * 0.10046665347629 * (2.6 - rij)^3
    end
    if 0 ≤ rij ≤ 2.7
        dpsi += -4 * 0.10054338881951 * (2.7 - rij)^3
    end
    if 0 ≤ rij ≤ 2.8
        dpsi += -4 * 0.099104582963213 * (2.8 - rij)^3
    end
    if 0 ≤ rij ≤ 3.0
        dpsi += -4 * 0.090086286376778 * (3.0 - rij)^3
    end
    if 0 ≤ rij ≤ 3.4
        dpsi += -4 * 0.0073022698419468 * (3.4 - rij)^3
    end
    if 0 ≤ rij ≤ 4.2
        dpsi += -4 * 0.014583614223199 * (4.2 - rij)^3
    end
    if 0 ≤ rij ≤ 4.8
        dpsi += 4 * 0.0010327381407070 * (4.8 - rij)^3
    end
    if 0 ≤ rij ≤ 5.6
        dpsi += -4 * 0.0073219994475288 * (5.6 - rij)^3
    end
    if 0 ≤ rij ≤ 6.5
        dpsi += -4 * 0.0095726042919017 * (6.5 - rij)^3
    end
    
    return -dpsi * vrij / rij
end

function EMAAl2_Phi_gradient(rho::Float64)::Float64
    dPhi = -0.5 / sqrt(rho)
    
    if rho >= 16
        dPhi += 4 * 3.5025051308271e-4 * (rho - 16)^3 -
                5 * 6.9606881126760e-5 * (rho - 16)^4 +
                6 * 3.5717262505601e-6 * (rho - 16)^5
    end
    
    if rho >= 24
        dPhi += 4 * -1.8828783364689e-3 * (rho - 24)^3 +
                5 * 1.1303817967915e-6 * (rho - 24)^4 -
                6 * 8.0158124278034e-6 * (rho - 24)^5
    end
    
    if rho >= 30
        dPhi += 4 * 2.7235851330782e-4 * (rho - 30)^3 +
                5 * 4.8914043715808e-5 * (rho - 30)^4 +
                6 * 4.7226737586655e-6 * (rho - 30)^5
    end
    
    return dPhi
end


function EMAAl2_rhoi(cell::UnitCell,i;ct::Float64=6.5)
    natom=length(cell.atoms)
    rhoi=0.0
    for j in 1:natom
        if j!=i
            rij=getrij(cell,i,j)
            nrij=norm(rij)
            if nrij<ct
                # println("$i,$j,$rij,$(EMACu_psi(nrij))")
                rhoi+=EMAAl2_psi(nrij)
            end
        end
    end
    return rhoi
end

# function EMAAl2_rhoi(cell::UnitCell, i; ct::Float64=6.5)
#     natom = length(cell.atoms)
#     rhoi = Atomic{Float64}(0.0)  # 使用原子类型保证线程安全

#     Threads.@threads for j in 1:natom
#         if j != i
#             rij = getrij(cell, i, j)
#             nrij = norm(rij)
#             if nrij < ct
#                 # 使用原子操作进行累加
#                 atomic_add!(rhoi, EMAAl2_psi(nrij))
#             end
#         end
#     end
#     return Float64(rhoi[]) 
# end

function embedding_energyAl2(cell::UnitCell)
    natom=length(cell.atoms)
    E=0.0
    for i in 1:natom
        rhoi=EMAAl2_rhoi(cell,i)
        E+=EMAAl2_Phi(rhoi)
    end
    return E
end

#这里多线程就会变慢不知道为什么
# function embedding_energyAl2(cell::UnitCell)
#     natom = length(cell.atoms)
#     # 如果只有一个线程，使用普通的累加
#     if Threads.nthreads() == 1
#         E = 0.0
#         for i in 1:natom
#             rhoi = EMAAl2_rhoi(cell, i)
#             E += EMAAl2_Phi(rhoi)
#         end
#         return E
#     else
#         # 如果线程数大于1，使用原子累加
#         E = Atomic{Float64}(0.0)
#         Threads.@threads for i in 1:natom
#             rhoi = EMAAl2_rhoi(cell, i)
#             atomic_add!(E, EMAAl2_Phi(rhoi))  # 使用原子累加
#         end
#         return Float64(E[])  # 返回普通浮点数
#     end
# end


function embedding_forceAl2ij(cell::UnitCell,i::Int,j::Int;ct::Float64=6.0)::SVector{3,Float64}
    rij=getrij(cell,i,j)
    Fij=SVector{3,Float64}(0.0, 0.0, 0.0)
    if norm(rij)<ct
        rhoi=EMAAl2_rhoi(cell,i)
        rhoj=EMAAl2_rhoi(cell,j)
        Fij=-(EMAAl2_Phi_gradient(rhoi)+EMAAl2_Phi_gradient(rhoj))*EMAAl2_psi_gradient(rij)
    end
    return Fij
end


function embedding_forceAl2ij(cell::UnitCell,i::Int,j::Int;ct::Float64=6.0)::SVector{3,Float64}
    rij=getrij(cell,i,j)
    Fij=SVector{3,Float64}(0.0, 0.0, 0.0)
    if norm(rij)<ct
        rhoi=EMAAl2_rhoi(cell,i)
        rhoj=EMAAl2_rhoi(cell,j)
        Fij=-(EMAAl2_Phi_gradient(rhoi)+EMAAl2_Phi_gradient(rhoj))*EMAAl2_psi_gradient(rij)
    end
    return Fij
end

function embedding_forceAl2i(cell::UnitCell, i::Int)::SVector{3, Float64}
    natom = length(cell.atoms)
    Fi = SVector{3, Float64}(0.0, 0.0, 0.0)
    for j in 1:natom
        if j != i
            Fi += embedding_forceAl2ij(cell, i, j)
        end
    end
    return Fi
end


function EMAAl1_phi(rij::Float64)::Float64
    phi = 0.0
    if 1.5 ≤ rij ≤ 2.3
        phi += exp(0.65196946237834 + 7.6046051582736 * rij - 5.8187505542843 * rij^2 + 1.0326940511805 * rij^3)
    end

    if 2.3 < rij ≤ 3.2
        phi += 13.695567100510 * (3.2 - rij)^4 - 44.514029786506 * (3.2 - rij)^5 + 
               95.853674731436 * (3.2 - rij)^6 - 83.744769235189 * (3.2 - rij)^7 + 
               29.906639687889 * (3.2 - rij)^8
    end
    
    if 2.3 < rij ≤ 4.8
        phi += -2.3612121457801 * (4.8 - rij)^4 + 2.5279092055084 * (4.8 - rij)^5 - 
               3.3656803584012 * (4.8 - rij)^6 + 0.94831589893263 * (4.8 - rij)^7 - 
               0.20965407907747 * (4.8 - rij)^8
    end

    if 2.3 < rij ≤ 6.5
        phi += 0.24809459274509 * (6.5 - rij)^4 - 0.54072248340384 * (6.5 - rij)^5 + 
               0.46579408228733 * (6.5 - rij)^6 - 0.18481649031556 * (6.5 - rij)^7 + 
               0.028257788274378 * (6.5 - rij)^8
    end
    return phi
end

function EMAAl1_psi(rij::Float64)::Float64
    psi = 0.0
    if 0 ≤ rij ≤ 2.5
        psi += 0.00019850823042883 * (2.5 - rij)^4
    end
    if 0 ≤ rij ≤ 2.6
        psi += 0.10046665347629 * (2.6 - rij)^4
    end
    if 0 ≤ rij ≤ 2.7
        psi += 0.10054338881951 * (2.7 - rij)^4
    end
    if 0 ≤ rij ≤ 2.8
        psi += 0.099104582963213 * (2.8 - rij)^4
    end
    if 0 ≤ rij ≤ 3.0
        psi += 0.090086286376778 * (3.0 - rij)^4
    end
    if 0 ≤ rij ≤ 3.4
        psi += 0.0073022698419468 * (3.4 - rij)^4
    end
    if 0 ≤ rij ≤ 4.2
        psi += 0.014583614223199 * (4.2 - rij)^4
    end
    if 0 ≤ rij ≤ 4.8
        psi += -0.0010327381407070 * (4.8 - rij)^4
    end
    if 0 ≤ rij ≤ 5.6
        psi += 0.0073219994475288 * (5.6 - rij)^4
    end
    if 0 ≤ rij ≤ 6.5
        psi += 0.0095726042919017 * (6.5 - rij)^4
    end
    return psi
end

function EMAAl1_Phi(rho::Float64)::Float64
    Phi = -sqrt(rho)
    if rho >= 16
        Phi += -6.1596236428225e-5 * (rho - 16)^4 + 1.4856817073764e-5 * (rho - 16)^5 - 
               1.4585661621587e-6 * (rho - 16)^6 + 7.2242013524147e-8 * (rho - 16)^7 - 
               1.7925388537626e-9 * (rho - 16)^8 + 1.7720686711226e-11 * (rho - 16)^9
    end
    return Phi
end

function EMAAl1_phi_gradient(vrij::SVector{3,Float64})::SVector{3,Float64}
    dphi = 0.0
    rij = norm(vrij)
    if 1.5 ≤ rij ≤ 2.3
        dphi += (7.6046051582736 - 2 * 5.8187505542843 * rij + 3 * 1.0326940511805 * rij^2) *
                exp(0.65196946237834 + 7.6046051582736 * rij - 5.8187505542843 * rij^2 + 1.0326940511805 * rij^3)
    end
    if 2.3 < rij ≤ 3.2
        dphi += -13.695567100510 * 4 * (3.2 - rij)^3 + 44.514029786506 * 5 * (3.2 - rij)^4 - 
                 95.853674731436 * 6 * (3.2 - rij)^5 + 83.744769235189 * 7 * (3.2 - rij)^6 - 
                 29.906639687889 * 8 * (3.2 - rij)^7
    end
    if 2.3 < rij ≤ 4.8
        dphi += 9.4448485831204 * (4.8 - rij)^3 - 12.639542477542 * (4.8 - rij)^4 + 
                14.2061942111952 * (4.8 - rij)^5 - 4.8664185162258 * (4.8 - rij)^6 + 
                1.6767526327748 * (4.8 - rij)^7
    end
    if 2.3 < rij ≤ 6.5
        dphi += -0.99237837098064 * (6.5 - rij)^3 + 2.7036124170176 * (6.5 - rij)^4 - 
                 2.7957644941569 * (6.5 - rij)^5 + 0.89767416515134 * (6.5 - rij)^6 - 
                 0.22460401982302 * (6.5 - rij)^7
    end
    return -dphi * vrij / rij
end

function EMAAl1_psi_gradient(vrij::SVector{3,Float64})::SVector{3,Float64}
    rij = norm(vrij)
    dpsi = 0.0
    if 0 ≤ rij ≤ 2.5
        dpsi += -4 * 0.00019850823042883 * (2.5 - rij)^3
    end
    if 0 ≤ rij ≤ 2.6
        dpsi += -4 * 0.10046665347629 * (2.6 - rij)^3
    end
    if 0 ≤ rij ≤ 2.7
        dpsi += -4 * 0.10054338881951 * (2.7 - rij)^3
    end
    if 0 ≤ rij ≤ 2.8
        dpsi += -4 * 0.099104582963213 * (2.8 - rij)^3
    end
    if 0 ≤ rij ≤ 3.0
        dpsi += -4 * 0.090086286376778 * (3.0 - rij)^3
    end
    if 0 ≤ rij ≤ 3.4
        dpsi += -4 * 0.0073022698419468 * (3.4 - rij)^3
    end
    if 0 ≤ rij ≤ 4.2
        dpsi += -4 * 0.014583614223199 * (4.2 - rij)^3
    end
    if 0 ≤ rij ≤ 4.8
        dpsi += 4 * 0.0010327381407070 * (4.8 - rij)^3
    end
    if 0 ≤ rij ≤ 5.6
        dpsi += -4 * 0.0073219994475288 * (5.6 - rij)^3
    end
    if 0 ≤ rij ≤ 6.5
        dpsi += -4 * 0.0095726042919017 * (6.5 - rij)^3
    end
    return -dpsi * vrij / rij
end

function EMAAl1_Phi_gradient(rho::Float64)::Float64
    dPhi = -0.5 / sqrt(rho)
    if rho >= 16
        dPhi += 4 * (-6.1596236428225e-5) * (rho - 16)^3 + 5 * (1.4856817073764e-5) * (rho - 16)^4 - 
                6 * (1.4585661621587e-6) * (rho - 16)^5 + 
                7 * (7.2242013524147e-8) * (rho - 16)^6 - 
                8 * (1.7925388537626e-9) * (rho - 16)^7 + 
                9 * (1.7720686711226e-11) * (rho - 16)^8
    end
    return dPhi
end



function EMAAl1_rhoi(cell::UnitCell,i;ct::Float64=6.5)
    natom=length(cell.atoms)
    rhoi=0.0
for j in 1:natom
        if j != i
            rij = getrij(cell, i, j)
            nrij = norm(rij)
            if nrij < ct
            
                rhoi+=EMAAl1_psi(nrij)
            end
        end
    end
    return rhoi
end


function embedding_energyAl1(cell::UnitCell)
    natom=length(cell.atoms)
    E=0.0
    for i in 1:natom
        rhoi=EMAAl1_rhoi(cell,i)
        E+=EMAAl1_Phi(rhoi)
    end
    return E
end


function embedding_forceAl1ij(cell::UnitCell,i::Int,j::Int;ct::Float64=6.0)::SVector{3,Float64}
    rij=getrij(cell,i,j)
    Fij=SVector{3,Float64}(0.0, 0.0, 0.0)
    if norm(rij)<ct
        rhoi=EMAAl1_rhoi(cell,i)
        rhoj=EMAAl1_rhoi(cell,j)
        Fij=-(EMAAl1_Phi_gradient(rhoi)+EMAAl1_Phi_gradient(rhoj))*EMAAl1_psi_gradient(rij)
    end
    return Fij
end

function embedding_forceAl1i(cell,i::Int)::SVector{3,Float64}
    natom=length(cell.atoms)

    Fi=SVector{3,Float64}(0.0, 0.0, 0.0)
    for j in 1:natom
        if j!=i
            Fi+=embedding_forceAl1ij(cell,i,j)
        end
    end
    return Fi 
end


end