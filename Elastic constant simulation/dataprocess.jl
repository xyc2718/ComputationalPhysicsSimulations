"""
code to process log file 
@author xyc
@email:22307110070m.fudan.edu.cn
"""
projectpath = "output\\AdHvRk3_Al2_222_300K_celltemp0"
ifgr=false
generate_frame=false
function fsel(i::Int)
    return true
end



using StaticArrays
using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using LsqFit
include("src\\Elastic.jl")
using .Elastic
using FFMPEG
using DelimitedFiles
using Plots
using JLD2
using Statistics


datapath=projectpath*"\\Log.txt"
figpath = replace(datapath, r"\.txt$" => ".png")
cellpath=replace(datapath, r"Log\.txt$" => "DumpCell.JLD2")
framepath=replace(datapath, r"Log\.txt$" => "Frames")
propertypath = replace(datapath, r"\\Log.txt$" => "\\Logproperty.txt")
data = readdlm(datapath);

Plots.plot(
Plots.plot(data[:,1],data[:,2],label="temperature",xlabel="timestep",ylabel="T",title="temperature-time"),
Plots.plot(data[:,1],data[:,3],label="Pressure",xlabel="timestep",ylabel="P",title="Pressure-time"),
Plots.plot(data[:,1],data[:,4],label="Volume",xlabel="timestep",ylabel="V",title="Volume-time"),
Plots.plot(data[:,1],data[:,5],label="Pv",xlabel="timestep",ylabel="Pv",title="Pv-time"),
size=(800,600), dpi=1000
)
savefig(figpath)


maxt = length(data[:,1])
bg = Int(floor(maxt / 4))
mean_T = mean(data[bg:maxt, 2])
mean_P = mean(data[bg:maxt, 3])
mean_V = mean(data[bg:maxt, 4])
mean_Pv = mean(data[bg:maxt, 5])

std_T = std(data[bg:maxt, 2])
std_P = std(data[bg:maxt, 3])
std_V = std(data[bg:maxt, 4])
std_Pv = std(data[bg:maxt, 5])

# 打印结果到控制台
println("<T> = ", mean_T)
println("<P> = ", mean_P)
println("<V> = ", mean_V)
println("<Pv> = ", mean_Pv)
println("std T = ", std_T)
println("std P = ", std_P)
println("std V = ", std_V)
println("std Pv = ", std_Pv)

# 打开文件并写入内容
open(propertypath, "w") do file
    write(file, "<T> = $mean_T\n")
    write(file, "<P> = $mean_P\n")
    write(file, "<V> = $mean_V\n")
    write(file, "<Pv> = $mean_Pv\n")
    write(file, "std T = $std_T\n")
    write(file, "std P = $std_P\n")
    write(file, "std V = $std_V\n")
    write(file, "std Pv = $std_Pv\n")
end

println("Results written to $propertypath")


if generate_frame || ifgr
    println("Loading cell data...")
    # datacell = JLD2.jldopen(cellpath, "r",mmap=true) do file
    #     Dict(name => read(file, name) for name in keys(file))
    # end
    datacell=JLD2.load(cellpath)
end

if generate_frame
    println("Generating frames...")
    outputfold=framepath
    sorted_keys = sort(collect(keys(datacell)), by = x -> parse(Int, split(x, "_")[2]))
    isdir(outputfold) || mkpath(outputfold)
    for key in sorted_keys
        i = parse(Int, split(key, "_")[2])
        if fsel(i) 
            fig=visualize_unitcell_atoms(datacell[key])
            save(joinpath(outputfold, "frame$i.jpg"), fig)
            println("frame$i.jpg saved")
        end
    end
end

if ifgr
    println("Calculating g(r)...")
    sorted_keys = sort(collect(keys(datacell)), by = x -> parse(Int, split(x, "_")[2]))
    grfigpath=projectpath*"\\gr.png"
    grpath=projectpath*"\\gr.txt"
    n=500
    rmin=0.5
    rmax=maximum(datacell["cell_1"].lattice_vectors*(datacell["cell_1"].copy)*2)*1.5
    rl=range(rmin,stop=rmax,length=n)
    gr=zeros(n)
    dr=(rmax-rmin)/n
    cell0=datacell[sorted_keys[1]]
    natom=length(cell0.atoms)
    for key in sorted_keys
    dcell=datacell[key]
    for i in 1:natom
        for j in i+1:natom
                rij=getrij(dcell,i,j)
                r=norm(rij)
                if r<rmax
                    k=Int(floor((r-rmin)/dr))+1
                    gr[k]+=1
            end
        end
    end
    end
    gr=gr./(4*pi*rl.^2*dr);
    normgr=(sum(gr)*dr-0.5*gr[1]*dr-0.5*gr[end]*dr)
    Plots.plot(rl.+dr/2,gr./normgr, bins=n, xlabel="r", ylabel="g(r)", title="Pair Correlation Function", label="g(r)",dpi=800)
    savefig(grfigpath)
    open(grpath,"w") do file
        writedlm(file,[rl,gr./normgr])
    end
    println("g(r) calculated and saved to $grfigpath")
end
