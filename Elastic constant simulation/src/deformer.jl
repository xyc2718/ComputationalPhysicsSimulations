module Deformer
using Distributions
using StaticArrays
using LinearAlgebra
using ..Model

export deform_mat, deform_cell!,FT2sigma,DF2elastic

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

    
function deform_cell!(fcell::UnitCell,deform_mat::Matrix{Float64},interaction::Interaction)
    fcell.lattice_vectors*=deform_mat
    update_rmat!(fcell)
    update_fmat!(fcell,interaction)
end

function FT2sigma(force_tensor::Matrix{T}) where T
    sigma = zeros(6)
    for i in 1:3
        sigma[i] = force_tensor[i, i]
    end
    sigma[4] = force_tensor[2, 3]
    sigma[5] = force_tensor[1, 3]
    sigma[6] = force_tensor[1, 2]
    return sigma
end

function FT2sigma(force_tensor_vec::Vector{Any})
    force_tensor =[Float64.(row) for row in force_tensor_vec] |> hcat
    return FT2sigma(Matrix(hcat(force_tensor...)))
end

FT2sigma(mat::SMatrix{3,3,T}) where T=FT2sigma(Matrix{T}(mat))


function DF2elastic(dmat::Matrix{T}) where T
    emat=dmat*transpose(dmat)-Matrix{Float64}(I, 3, 3)
    e=zeros(6)
    for i in 1:3
        e[i]=emat[i,i]
    end
    e[4]=2*emat[2,3]
    e[5]=2*emat[1,3]
    e[6]=2*emat[1,2]
    return e./2
end

DF2elastic(flag::Int,delta::Float64)=DF2elastic(deform_mat(flag,delta))

end