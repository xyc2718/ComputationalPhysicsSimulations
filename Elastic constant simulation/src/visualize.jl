module Visualize
using GLMakie 
using ..Model
using LaTeXStrings
using Printf
using JSON
export visualize_unitcell_atoms,visualize_unitcell_atoms0,matrix_to_latex,read_json,read_config


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
function visualize_unitcell_atoms(cell::UnitCell;markersize=10,veccolor=:blue,linewith=0.1)::Figure
    fig =GLMakie.Figure(size = (800, 600))
    ax = GLMakie.Axis3(fig[1, 1], title = "Visualization of Atoms in the Unit Cell", 
               xlabel = "X", ylabel = "Y", zlabel = "Z")
    M=cell.lattice_vectors
    for atom in cell.atoms
        p=M*atom.position
        cni=atom.cn
        GLMakie.scatter!(ax,p..., color = color_map(cni), markersize = markersize)
    end
    
    # 绘制晶格向量（晶胞的边缘）
    origin=GLMakie.Point3f0(0,0,0)
    for (k,vec) in enumerate(eachcol(cell.lattice_vectors))
        vc=GLMakie.Point3f0(vec*cell.copy[k])
        GLMakie.arrows!(ax, [origin], [origin + vec], color = veccolor, linewidth = linewith)
    end
    rg=maximum(cell.lattice_vectors*cell.copy)
    rgmin=minimum(cell.lattice_vectors*cell.copy)
    xlims!(ax, -rg, rg)
    ylims!(ax, -rg, rg)
    zlims!(ax, -rg, rg)
    return fig
end

"""
可视化晶胞原子,用于测试原子因某些原因跑出周期范围,并可标记原子序号,用于调试
:param cell:UnitCell
:param markersize:Int 标记大小
:param iftext:Bool 是否标记原子序号
"""
function visualize_unitcell_atoms0(cell::UnitCell;markersize=10,iftext::Bool=false)::Figure
    fig =GLMakie.Figure(size = (800, 600))
    ax = GLMakie.Axis3(fig[1, 1], title = "Visualization of Atoms in the Unit Cell", 
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
            text!(ax,string(i),position=Point3f(p))
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


end