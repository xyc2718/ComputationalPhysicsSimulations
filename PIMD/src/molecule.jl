module MoleculeUtils
using ..Model
using StaticArrays
export  mapCell2Molecue
"""
Map atoms and molecule to the pos of atoms in cell
"""
function mapCell2Molecue(cell::UnitCell, molecule::Molecule)
    matoms=Vector{Atom}([])
    connection=Vector{Vector{Int}}([])
    id=1
    for catom in cell.atoms
        r0=catom.position
        dr=-molecule.atoms[1].position+r0
        cni=Vector{Int}(id:id+length(molecule.atoms)-1)
        push!(connection,cni)
        for ma in molecule.atoms            
            push!(matoms,deepcopy(ma))
            matoms[end].position+=dr
            id=id+1
        end
    end
    mcell=deepcopy(cell)
    mcell.atoms=matoms
    mcell.fmat=Vector{SVector{3,Float64}}(undef,length(mcell.atoms))
    mcell.rmat=Matrix{SVector{3,Float64}}(undef,length(mcell.atoms),length(mcell.atoms))
    mol=Molecule(connection,molecule.atoms)
    return mcell,mol

end

end