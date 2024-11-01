using StaticArrays
using LinearAlgebra
# using Makie
using GLMakie 
using LsqFit
include("src\\Elastic.jl")
using .Elastic
using DelimitedFiles
using Distributions
using Statistics
using Plots
using Test
using IterTools
using BenchmarkTools
using DelimitedFiles  # 用于CSV写入
using JSON            # 用于JSON格式

using LaTeXStrings
atom_positions = [
    Vector([0.0, 0.0, 0.0]),
    Vector([1.0, 1.0, 1.0]),
    Vector([1.0, 0.0, 0.0]),
    Vector([0.0, 1.0, 0.0]),
    Vector([0.0, 0.0, 1.0]),
    Vector([0.0, 1.0, 1.0]),
    Vector([1.0, 0.0, 1.0]),
    Vector([1.0, 1.0, 0.0]),
    Vector([0.5, 0.5, 0.0]),
    
    Vector([0.5, 0.0, 0.5]),
    Vector([0.0, 0.5, 0.5]),
    Vector([1.0, 0.5, 0.5]),
    Vector([0.5, 1.0, 0.5]),
    Vector([0.5, 0.5, 1.0]),
    Vector([0.75, 0.25, 0.25]),
    Vector([0.25, 0.75, 0.25]),
    Vector([0.25, 0.25, 0.75]),
    Vector([0.75, 0.75, 0.75]),
]

kb=8.617332385e-5 #eV/K
amuM=1.03642701e-4 #[m]/amu
Mc=12 #amu
lattice_constant = 3.566 #A

# 定义FCC晶胞的基矢量
lattice_vectors = collect((Matrix([
    lattice_constant 0.0 0.0; #a1
    0.0 lattice_constant 0.0; #a2
    0.0 0.0 lattice_constant] #a3
))')

atoms = [Atom(pos,Mc*amuM) for pos in atom_positions]


cell=UnitCell(lattice_vectors,atoms)
cpcell=copycell(cell,1,1,1)
fcell=filtercell(cpcell)




projectname="deform_dimaond_111"
Comment="Deform Diamond 111 not gradientDescent"
cp=[1,1,1]
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
deltalist=range(-0.005,0.005,length=20)
# deltalist=(-0.001,0.001)
flaglist=[1,2,3,4,5,6]
checkstep=10
ap=0.001
tol=1e-8
maxiter=500

iniminimizerange=[3.4,3.7]
minpoint=1000

cell=UnitCell(lattice_vectors,atoms)
cpcell=copycell(cell,cp...)
fcell=filtercell(cpcell)



sw=SW(SWenergy_Diamond,SWforcei_Diamond)
interaction=Interaction(v2Diamond,v2gradient_Diamond,a*sigma+0.1,0.1,sw)

natom=length(fcell.atoms)

minimizeEnergy!(fcell,interaction,rg=iniminimizerange,n=minpoint)


iniE=cell_energy(fcell,interaction)
iniltv=deepcopy(fcell.lattice_vectors)
println("Initial Energy: $iniE")
println("Initial Lattice Vectors: $iniltv")

basepath="Deformer_output\\$projectname"
if !isdir(basepath)
    mkpath(basepath)
    println("Directory $basepath created.\n")
else
    local counter = 1
    local newpath = basepath * "_$counter"
    while isdir(newpath)
        counter += 1
        newpath = basepath * "_$counter"
    end
    mkpath(newpath)
    println("Directory exists,new Directories $newpath created.\n")
    basepath=newpath
end

open("$basepath\\Config.txt", "w") do logfile
    write(logfile, "projectname=$projectname\n")
    write(logfile,"Method:gradientDescent,Interaction:SW")
    write(logfile, "$natom  atoms\n")
    write(logfile, "Md for Fcc Cu\n")
    write(logfile, "cpsize=$cp\n")
    write(logfile, "maxiter=$maxiter\n")
    write(logfile, "alpha=$ap\n")
    write(logfile, "tol=$tol\n")
    write(logfile, "checkstep=$checkstep\n")
    write(logfile, "ct=$ct\n")
    write(logfile, "deltalist=$deltalist\n")
    write(logfile, "flaglist=$flaglist\n")
    write(logfile, "iniminimizerange=$iniminimizerange\n")
    write(logfile, "minpoint=$minpoint\n")
    write(logfile, "Initial Energy: $iniE\n")
    write(logfile, "Initial Lattice Vectors: $iniltv\n")
    write(logfile,"Comment:$Comment")

end

open("$basepath\\Log.json", "w") do file
for flag in flaglist
    for delta in deltalist
        println("flag=$flag,delta=$delta")
        cell=deepcopy(fcell)
        dmat=deform_mat(flag,delta)
        deform_cell!(cell,dmat,interaction)
        El=[]
        El=gradientDescent!(cell,interaction,ap=ap,tol=tol,maxiter=maxiter,checktime=checkstep)
        forcetensor=force_tensor(cell,interaction)
        ltv=cell.lattice_vectors
        V=cell.Volume
        dUdh=dUdhij(cell,interaction)
        ft0=forcetensor-(dUdh*transpose(ltv))./V
    data = Dict(
        "flag" => flag,
        "delta"=> delta,
        "forcetensor0" => forcetensor,
        "dUdhij" => dUdh,
        "hij"=>ltv,
        "V"=>V,
        "forcetensor"=>ft0,
        "Eenergy"=>El
    )

    println(data)
    write(file, JSON.json(data))
    write(file, "\n")
    
    end
end

end
