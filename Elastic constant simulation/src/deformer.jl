module Deformer
using Distributions
using StaticArrays
using LinearAlgebra
using ..Model

export deform_mat, deform_cell!

function deform_mat(flag::Int, delta::Float64)
    if flag == 1
        return [1.0 + delta 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    elseif flag == 2
        return [1.0 0.0 0.0; 0.0 1.0 + delta 0.0; 0.0 0.0 1.0]
    elseif flag == 3
        return [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0 + delta]
    elseif flag == 4
        return [1.0 delta 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    elseif flag == 5
        return [1.0 0.0 delta; 0.0 1.0 0.0; 0.0 0.0 1.0]
    elseif flag == 6
        return [1.0 0.0 0.0; 0.0 1.0 delta; 0.0 0.0 1.0]
    else
        error("Invalid flag value. Must be an integer between 1 and 6.")
    end
end

    
function deform_cell!(fcell::UnitCell,deform_mat::Matrix{Float64})
    fcell.lattice_vectors*=deform_mat
end

end