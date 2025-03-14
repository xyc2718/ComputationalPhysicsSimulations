module Visualize
using Makie 
using ..Model
using ..PIMD
using LaTeXStrings
using LinearAlgebra
using Printf
using JSON
using StaticArrays
export visualize_unitcell_atoms,visualize_unitcell_atoms0,matrix_to_latex,read_json,read_config,visualize_beadcell,calculate_gr!,DataProcessor,RadialDistribution,Normalize_gr!,SpatialDistribution,calculate_pr!,Normalize_pr!,Trajectory,calculate_traj!,fix_traj!,clear_traj!

try
    using GLMakie
catch
    try
        using CairoMakie
    catch
        println("Failed to load GLMakie or CairoMakie")
end


# 定义一个颜色映射函数
function color_map(cn)
    colors = [:red, :green, :blue, :yellow, :purple, :orange, :cyan, :magenta]
    return colors[mod(cn - 1, length(colors)) + 1]
end


"""
可视化晶胞原子,并标注晶格向量
:param cell: UnitCell晶胞
:param markersize:Int 标记大小
:param veccolor: 向量颜色
:param linewith: 线宽
"""
function visualize_unitcell_atoms(cell::UnitCell;markersize=10,veccolor=:blue,linewith=0.1,sizelist::Vector{Float64}=Vector{Float64}(undef,0),colorlist::Vector{Int}=Vector{Int}(undef,0))::Figure
    fig =Makie.Figure(size = (800, 600))
    ax = Makie.Axis3(fig[1, 1], title = "Visualization of Atoms in the Unit Cell", 
               xlabel = "X", ylabel = "Y", zlabel = "Z")
    M=cell.lattice_vectors
    lsz=length(sizelist)
    csz=length(colorlist)
    for k in eachindex(cell.atoms)
        atom=cell.atoms[k]
        if k<=lsz
            ms=sizelist[k]*markersize
        else
            ms=markersize
        end
        if k<=csz
            cs=colorlist[k]
        else
            cs=atom.cn
        end
        p=M*atom.position
        Makie.scatter!(ax,p..., color = color_map(cs), markersize = ms)
    end
    
    # 绘制晶格向量（晶胞的边缘）
    origin=Makie.Point3f0(0,0,0)
    for (k,vec) in enumerate(eachcol(cell.lattice_vectors))
        vc=Makie.Point3f0(vec*cell.copy[k])
        Makie.arrows!(ax, [origin], [origin + vec], color = veccolor, linewidth = linewith)
    end
    rg1=maximum(cell.lattice_vectors*[cell.copy[1],0.0,0.0])
    rg2=maximum(cell.lattice_vectors*[0.0,cell.copy[2],0.0])
    rg3=maximum(cell.lattice_vectors*[0.0,0.0,cell.copy[3]])
    rgmin=minimum(cell.lattice_vectors*cell.copy)
    Makie.xlims!(ax, -rg1, rg1)
    Makie.ylims!(ax, -rg2, rg2)
    Makie.zlims!(ax, -rg3, rg3)
    return fig
end

"""
可视化晶胞原子,用于测试原子因某些原因跑出周期范围,并可标记原子序号,用于调试
:param cell:UnitCell
:param markersize:Int 标记大小
:param iftext:Bool 是否标记原子序号
"""
function visualize_unitcell_atoms0(cell::UnitCell;markersize=10,iftext::Bool=false)::Figure
    fig =Makie.Figure(size = (800, 600))
    ax = Makie.Axis3(fig[1, 1], title = "Visualization of Atoms in the Unit Cell", 
               xlabel = "X", ylabel = "Y", zlabel = "Z")
    M=cell.lattice_vectors
    for i in 1:length(cell.atoms)
        atom=cell.atoms[i]
        p=M*atom.position
        cni=atom.cn
        scatter!(ax,p..., color = color_map(cni), markersize = markersize)
    end
    if iftext
        for i in 1:length(cell.atoms)
            atom=cell.atoms[i]
            p=M*atom.position
            Makie.text!(ax,string(i),position=Point3f(p))
        end
    end

    return fig
end

"""
生成latex矩阵字符串
"""
function matrix_to_latex(matrix::Matrix{T};n::Int=3) where T
    rows, cols = size(matrix)
    latex_str = "\\left[\\begin{matrix}"
    for i in 1:rows
        for j in 1:cols
            latex_str *= @sprintf("%.*f", n, matrix[i, j])
            if j < cols
                latex_str *= " & "
            end
        end
        if i < rows
            latex_str *= " \\\\ "
        end
    end
    latex_str *= "\\end{matrix}\\right]"
    return L"%$latex_str"  # 生成LaTeXString格式
end

function read_json(filename)
    results = []
    open(filename, "r") do file
        for line in eachline(file)
            if length(line) <=5
                continue
            end
            data = JSON.parse(line)
            push!(results, data)
        end
    end
    return results
end

function read_config(path, keyword)
    lines= readlines(path)
    for line in lines
        if occursin(keyword, line)
            return strip(replace(line, "$keyword" => ""))
        end
    end
    return nothing
end

function visualize_beadcell(bdc::BeadCell;markersize=10,veccolor=:blue,linewith=0.01,alpha=0.5)
    fig =Makie.Figure(size = (800, 600))
    ax = Makie.Axis3(fig[1, 1], title = "Visualization of Atoms in the Unit Cell", 
    xlabel = "X", ylabel = "Y", zlabel = "Z")
    nbeads=bdc.nbeads
    for i in 1:nbeads
        cell=bdc.cells[i]
        M=cell.lattice_vectors
        for atom in cell.atoms
            p=M*atom.position
            cni=atom.cn
            Makie.scatter!(ax,p..., color =:red, markersize = markersize,alpha=alpha)
        end
        origin=Makie.Point3f0(0,0,0)
        for (k,vec) in enumerate(eachcol(cell.lattice_vectors))
            vc=Makie.Point3f0(vec*cell.copy[k])
            Makie.arrows!(ax, [origin], [origin + vec], color = veccolor, linewidth = linewith)
        end
        rg1=maximum(cell.lattice_vectors*[cell.copy[1],0.0,0.0])
        rg2=maximum(cell.lattice_vectors*[0.0,cell.copy[2],0.0])
        rg3=maximum(cell.lattice_vectors*[0.0,0.0,cell.copy[3]])
        rgmin=minimum(cell.lattice_vectors*cell.copy)
        Makie.xlims!(ax, -rg1, rg1)
        Makie.ylims!(ax, -rg2, rg2)
        Makie.zlims!(ax, -rg3, rg3)
    end
    return fig

end
abstract type DataProcessor end

mutable struct RadialDistribution<:DataProcessor
    type::String
    rmin::Float64
    rmax::Float64
    n::Int
    dr::Float64
    rl::Vector{Float64}
    rm::Vector{Float64}
    gr::Vector{Float64}
    ngr::Vector{Float64}
    Gr::Vector{Float64}
    function RadialDistribution(rmin::Float64,rmax::Float64,n::Int)
        dr=(rmax-rmin)/n
        rl=range(rmin,stop=rmax,length=n)
        rm=rl.+dr/2
        gr=zeros(n)
        ngr=zeros(n)
        Gr=zeros(n)
        new("RadialDistribution",rmin,rmax,n,dr,rl,rm,gr,ngr,Gr)
    end
end

function calculate_gr!(gr::RadialDistribution,cell::UnitCell)
    natom=length(cell.atoms)
    for i in 1:natom
        for j in i+1:natom
                rij=getrij(cell,i,j)
                r=norm(rij)
                if (r<gr.rmax)&&(r>gr.rmin)
                    k=Int(floor((r-gr.rmin)/gr.dr))+1
                    gr.Gr[k]+=1
                end
        end
    end
end


function calculate_gr!(gr::RadialDistribution,cell::UnitCell,i::Int,j::Int)
    natom=length(cell.atoms)
    
                rij=getrij(cell,i,j)
                r=norm(rij)
                if (r<gr.rmax)&&(r>gr.rmin)
                    k=Int(floor((r-gr.rmin)/gr.dr))+1
                    gr.Gr[k]+=1
                end

end

function calculate_gr!(gr::RadialDistribution,bdc::BeadCell)
    natom=length(bdc.cells[1].atoms)
    N=bdc.nbeads
    for i in 1:natom
        for j in i+1:natom
                rij=SVector{3,Float64}([0.0,0.0,0.0])
                for cell in bdc.cells
                rij+=getrij(cell,i,j)
                end
                rij=rij./N
                r=norm(rij)
                if (r<gr.rmax)&&(r>gr.rmin)
                    k=Int(floor((r-gr.rmin)/gr.dr))+1
                    gr.Gr[k]+=1
                end
        end
    end
end

function calculate_gr!(gr::RadialDistribution,bdc::BeadCell,i::Int,j::Int)
    natom=length(bdc.cells[1].atoms)
    N=bdc.nbeads
   
                rij=SVector{3,Float64}([0.0,0.0,0.0])
                for cell in bdc.cells
                rij+=getrij(cell,i,j)
                end
                rij=rij./N
                r=norm(rij)
                if (r<gr.rmax)&&(r>gr.rmin)
                    k=Int(floor((r-gr.rmin)/gr.dr))+1
                    gr.Gr[k]+=1
                end
    
end

function Normalize_gr!(gr::RadialDistribution)
    dr=gr.dr
    rm=gr.rm
    gr.gr=(gr.Gr)./(4*pi*(rm.^2).*dr);
    n=(sum(gr.gr)*dr-0.5*(gr.gr[1])*dr-0.5*(gr.gr[end])*dr)
    gr.ngr=gr.gr/n
end


mutable struct SpatialDistribution<:DataProcessor
    type::String
    lattice_vectors::Matrix{Float64}
    rmin::Vector{Float64}
    rmax::Vector{Float64}
    nx::Int
    ny::Int
    nz::Int
    dx::Float64
    dy::Float64
    dz::Float64
    xl::Vector{Float64}
    yl::Vector{Float64}
    zl::Vector{Float64}
    xm::Vector{Float64}
    ym::Vector{Float64}
    zm::Vector{Float64}
    pr::Array{Float64, 3}
    npr::Array{Float64, 3}
    function SpatialDistribution(cell::UnitCell,nx::Int,ny::Int,nz::Int;rmin::Vector{Float64}=[-1.0,-1.0,-1.0],rmax::Vector{Float64}=[1.0,1.0,1.0])
        xmin,ymin,zmin=rmin
        xmax,ymax,zmax=rmax
        dx=(xmax-xmin)/nx
        dy=(ymax-ymin)/ny
        dz=(zmax-zmin)/nz
        if nx==1
            xl=[xmin]
        else    
        xl=range(xmin,stop=xmax,length=nx)
        end
        
        if ny==1
            yl=[ymin]
        else
            yl=range(ymin,stop=ymax,length=ny)
        end
        if nz==1
            zl=[zmin]
        else
            zl=range(zmin,stop=zmax,length=nz)
        end
        xm=xl.+dx/2
        ym=yl.+dy/2
        zm=zl.+dz/2
        pr=zeros(nx,ny,nz)
        npr=zeros(nx,ny,nz)
        new("SpatialDistribution",cell.lattice_vectors,rmin,rmax,nx,ny,nz,dx,dy,dz,xl,yl,zl,xm,ym,zm,pr,npr)
    end
    function SpatialDistribution(bdc::BeadCell,nx::Int,ny::Int,nz::Int)
        SpatialDistribution(bdc.cells[1],nx,ny,nz)
    end
end

function calculate_pr!(pr::SpatialDistribution,cell::AbstractCell)
    xmin,ymin,zmin=pr.rmin
    natom=get_natom(cell)
    for i in 1:natom
            ri=get_position0(cell,i)
            if all(ri.<pr.rmax)&&all(ri.>pr.rmin)
                ix=Int(floor((ri[1]-xmin)/pr.dx))+1
                iy=Int(floor((ri[2]-ymin)/pr.dy))+1
                iz=Int(floor((ri[3]-zmin)/pr.dz))+1
                # println("iy=$iy,ri=$ri,ymin=$ymin")
                pr.pr[ix,iy,iz]+=1
            end
    end
end

function Normalize_pr!(pr::SpatialDistribution)
    n=(sum(pr.pr)*pr.dx*pr.dy*pr.dz)
    pr.npr=pr.pr/n
end

mutable struct Trajectory<:DataProcessor
    type::String
    rl::Matrix{Float64}
    vl::Matrix{Float64}
    maxstep::Int
    dt::Float64
    t::Int
    ti::Int
    savetimes::Int
    function Trajectory(maxstep::Int,dt::Float64=1.0)
        rl=zeros(3,maxstep)
        vl=zeros(3,maxstep)
        new("Trajectory",rl,vl,maxstep,dt,1,1,0)
    end
    function Trajectory(beginstep::Int,endstep::Int,samplesequence::Int,dt::Float64=1.0)
        maxstep=Int(floor((endstep-beginstep)/samplesequence))+1
        rl=zeros(3,maxstep)
        vl=zeros(3,maxstep)
        new("Trajectory",rl,vl,maxstep,dt*samplesequence,1,1,0)
    end
end

function calculate_traj!(tr::Trajectory,cell::AbstractCell,i::Int;PBC::Bool=false)
    # println(tr.t)
    tr.rl[:,tr.ti].=get_position(cell,i,PBC=PBC)
    tr.vl[:,tr.ti].=get_velocity(cell,i) 
    tr.t+=1
    tr.ti+=1
end

function clear_traj!(tr::Trajectory)
    # println(tr.t)
    tr.rl=zeros(3,tr.maxstep)
    tr.vl=zeros(3,tr.maxstep)
    tr.ti=1
    tr.savetimes+=1
end

function clear_traj!(trl::Vector{Trajectory})
    for tr in trl
        clear_traj!(tr)
    end
end

function fix_traj!(trl::Vector{Trajectory},cell::AbstractCell;PBC::Bool=false)
    natom=get_natom(cell)
    for i in 1:natom
        calculate_traj!(trl[i],cell,i,PBC=PBC)
    end
    
end

end