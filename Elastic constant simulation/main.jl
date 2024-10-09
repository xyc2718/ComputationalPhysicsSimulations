using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using LsqFit
include("Elastic.jl")
using .Elastic
using FFMPEG
using DelimitedFiles
using Base.Threads

tr=Threads.nthreads()
println("Threads: $tr")

#lj势能
function lj(r::Float64)
    return 4*(1/r^12-1/r^6)
end
function Flj(r::Vector{Float64})
    rn=norm(r)
    return 24*(2/rn^14-1/rn^8)*r
end

ct=5.0
interaction = Interaction(lj, Flj, ct, 0.1)

x=1:0.001:ct
y=interaction.cutenergy.(x)
atom_positions = [
    Vector([0.0, 0.0, 0.0]),
    Vector([0.0, 0.5, 0.5]),
    Vector([0.5, 0.0, 0.5]),
    Vector([0.5, 0.5, 0.0]),
    Vector([1.0, 0.0, 0.0]),
    Vector([0.0, 1.0, 0.0]),
    Vector([0.0, 0.0, 1.0]),
    Vector([0.5, 1.0, 0.5]),
    Vector([1.0, 0.5, 0.5]),
    Vector([0.5, 0.5, 1.0]),
    Vector([1.0, 0.0, 1.0]),
    Vector([1.0, 1.0, 0.0]),
    Vector([0.0, 1.0, 1.0]),
    Vector([1.0, 1.0, 1.0])
] 

el=Vector{Float64}([])
cl=1.3:0.02:2.5
cp=[3,3,3]
fl=Vector{Float64}([])

for lattice_constant in cl
    # 定义铜的FCC晶胞的基矢量
    lattice_vectors = (Matrix([
        lattice_constant 0.0 0.0; #a1
        0.0 lattice_constant 0.0; #a2
        0.0 0.0 lattice_constant] #a3
    ))
    # 创建铜的原子列表
    atoms = [Atom(pos) for pos in atom_positions]

    cell=UnitCell(lattice_vectors,atoms)
    cpcell=copycell(cell,cp...)
    fcell=filtercell(cpcell)
    en=cell_energy0(fcell,interaction,ifnormalize=false)
    push!(el,en)
    fn=sum((force_tensor(fcell,interaction)).^2)*fcell.Volume*27
    push!(fl,fn)
end

minlt=cl[argmin(el)]
println("minlt=$minlt")
lattice_vectors = (Matrix([
    minlt 0.0 0.0; #a1
    0.0 minlt 0.0; #a2
    0.0 0.0 minlt] #a3
))
atoms = [Atom(pos) for pos in atom_positions]

fcell=initcell(10.0,10.0,atoms,interaction,cp=[3,3,3],Prg=[0.03,8])
println("energt=",cell_energy0(fcell,interaction,ifnormalize=false))
println("forcetensor=",(force_tensor(fcell,interaction)))
println("pressure=",pressure_int(fcell,interaction))
fig=visualize_unitcell_atoms(fcell)
display(fig)

thermostat = Thermostat(10.0, 100000.0, 0.0, 0.0)
barostat=Barostat(10.0,100000.0,fcell.Volume,0.0)
println(pressure_int(fcell,interaction))
z=cell2z(fcell,thermostat,barostat);

println("\nbegin\n")
tpcell=deepcopy(fcell)
open("data_z.txt", "a") do io
    z=cell2z(tpcell,thermostat,barostat)
    for i in 1:10000 
        RK3_step!(z,0.001,tpcell,interaction,thermostat,barostat)    

        # 将 z 写入文件
        writedlm(io, z')
        write(io, "\n")
        if i%10==0
            p=pressure_int(tpcell,interaction)
            T=cell_temp(tpcell)

            println("step=$i,pressure=$p,temperature=$T")

        end
    end  
end         
println("\nprocess done")

