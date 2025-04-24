using StaticArrays
using LinearAlgebra
# using Makie
using LsqFit
include("src\\Elastic.jl")
using .Elastic
using JLD2



trpath="output\\ICE_111_200K_13\\tr.JLD2"
outputfold="vc-T-200"
path="\\vc_B8.xyz"
if !isdir(outputfold)
    mkdir(outputfold)
end
traj=load(trpath,"tr")
maxstep=2500
natom=length(traj)
open(outputfold*path,"w") do f
for i in 1:maxstep
    write(f,"$natom\n")
    write(f,"# CELL(abcABC):   18.64388    18.64388    18.64388    90.00000    90.00000    90.00000  Step:           $(i)  Bead:       0 v_centroid{atomic_unit}  cell{atomic_unit}\n")
    for k in 1:length(traj)
        if mod(k,3)==1
            write(f,"\tO $(traj[k].vl[1,i])\t$(traj[k].vl[2,i])\t$(traj[k].vl[3,i])\n")
        else
            write(f,"\tH $(traj[k].vl[1,i])\t$(traj[k].vl[2,i])\t$(traj[k].vl[3,i])\n")
        end
    end

end
end

println("save $trpath to $(outputfold*path) done")