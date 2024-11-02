module Wave 
using ..Model

export applyWavePerturbation!

"""
Apply wave perturbation to the atoms in the unit cell
:param fcell: UnitCell
:param interaction: Interaction
:param waveVector: Vector{Float64}=[0.0,0.0,1.0] the perturbation apply to the atoms(position)
:param bd: Vector{Int}=[1,1,0] the bound of the atoms to apply the perturbation
"""
function applyWavePerturbation!(fcell::UnitCell,interaction::Interaction,waveVector::Vector{Float64}=[0.0,0.0,0.1];bd::Vector{Vector{Int}}=[[1,1,0],[1,1,1],[1,0,0]])
    for bd0 in bd
        for i in 1:length(fcell.atoms)
            if fcell.atoms[i].bound==bd0
                fcell.atoms[i].position.+=waveVector
            end
        end
    end
    update_rmat!(fcell)
    update_fmat!(fcell,interaction)
    return nothing
end
end