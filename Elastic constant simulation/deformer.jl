module Deformer
using Distributions
using StaticArrays
using LinearAlgebra
using GLMakie 
using ..Model
using Base.Threads
using JLD2



function deform_mat(flag::Int, delta::Float64)
    if flag == 1
        return [1 + delta 0 0; 0 1 0; 0 0 1]
    elseif flag == 2
        return [1 0 0; 0 1 + delta 0; 0 0 1]
    elseif flag == 3
        return [1 0 0; 0 1 0; 0 0 1 + delta]
    elseif flag == 4
        return [1 delta 0; 0 1 0; 0 0 1]
    elseif flag == 5
        return [1 0 delta; 0 1 0; 0 0 1]
    elseif flag == 6
        return [1 0 0; 0 1 delta; 0 0 1]
    else
        error("Invalid flag value. Must be an integer between 1 and 6.")
    end
end
    
function deform_cell(fcell::UnitCell,deform_mat::Float64)4
    fcell.a1 = fcell.a1 * deform_mat
    fcell.a2 = fcell.a2 * deform_mat
    fcell.a3 = fcell.a3 * deform_mat
    return fcell
end



end