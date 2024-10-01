module Model
    
using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 

export Atom, UnitCell, copycell, visualize_unitcell_atoms, Interaction


"""
原子类型
:param element: 元素名称
:param position: 原子位置
:param cn: 配位数
"""
struct Atom
    element::String
    position::Vector{Float64}
    cn::Int
    function Atom(element,position,cn)
        new(element,position,cn)
    end
    function Atom(element,position)
        new(element,position,1)
    end
end



"""
晶胞类型
:param lattice_vectors: 晶格矢量
:param atoms: 原子数组
:param copy: 晶胞复制次数,default=[1,1,1]
:param Volume: 晶胞体积,default=det(lattice_vectors)*copy[1]*copy[2]*copy[3]
"""
struct UnitCell
    lattice_vectors::Matrix{Float64}
    atoms::Vector{Atom}
    copy::Vector{Int}
    Volume::Float64
    function UnitCell(lattice_vectors::Matrix{Float64}, atoms::Vector{Atom}, copy::Vector{Int})
        v=copy[1]*copy[2]*copy[3]*det(lattice_vectors)
        new(lattice_vectors, atoms,copy,v)
    end
    function UnitCell(lattice_vectors::Matrix{Float64}, atoms::Vector{Atom})
        new(lattice_vectors, atoms, Vector([1,1,1]))
    end
    function UnitCell(lattice_vectors::Adjoint{Float64, Matrix{Float64}}, atoms::Vector{Atom})
        new(lattice_vectors, atoms, Vector([1,1,1]))
    end
end


"""
复制晶胞
:param cell: 晶胞
:param a,b,c: 复制次数
:param tol: 误差,default=0.001,用于计算配位数
"""
function copycell(cell::UnitCell,a::Int,b::Int,c::Int,tol::Float64=0.001)::UnitCell
    atoms=Vector{Atom}([])
    pl=Vector{Vector{Float64}}([])
    for atom in cell.atoms
        for i in 0:a-1
            for j in 0:b-1
                for k in 0:c-1
                    newp=atom.position+Vector([i,j,k])
                    if newp in pl
                        continue
                    end
                    if (newp==[0.0,0.0,0.0]||newp==[1.0*a,1.0*b,1.0*c])
                        cn=8
                    elseif count(x -> abs(x) < tol, newp)==2
                        cn=4
                    elseif count(x -> abs(x) < tol, newp)==1
                        cn=2
                    else
                        cn=1 
                    end

                    push!(atoms,Atom(atom.element,newp,cn))
                    push!(pl,newp)
                end
            end
        end      
    end
    
    return UnitCell(cell.lattice_vectors,atoms,Vector([a,b,c]))
end


"""
可视化晶胞原子
"""
function visualize_unitcell_atoms(cell::UnitCell,atomcolor=:red,markersize=10,veccolor=:blue,linewith=0.1)::Figure
    fig =GLMakie.Figure(size = (800, 600))
    ax = GLMakie.Axis3(fig[1, 1], title = "Visualization of Atoms in the Unit Cell", 
               xlabel = "X", ylabel = "Y", zlabel = "Z")
    M=cell.lattice_vectors
    for atom in cell.atoms
        p=M*atom.position
        GLMakie.scatter!(ax,p..., color = atomcolor, markersize = markersize)
    end
    
    # 绘制晶格向量（晶胞的边缘）
    origin=GLMakie.Point3f0(0,0,0)
    for (k,vec) in enumerate(eachcol(cell.lattice_vectors))
        vc=GLMakie.Point3f0(vec*cell.copy[k])
        GLMakie.arrows!(ax, [origin], [origin + vec], color = veccolor, linewidth = linewith)
    end
    rg=maximum(cell.lattice_vectors*cell.copy)
    xlims!(ax, 0, rg)
    ylims!(ax, 0, rg)
    zlims!(ax, 0, rg)
    return fig
end

"""
相互作用类型,通过二次函数添加截断于cutoff-cutrg - cutoff
:param energe: 势能函数
:param force: 力函数
:param cutenerge: 截断势能函数
:param cutforce: 截断力函数
:param cutoff: 截断距离
:param cutrg: 截断范围
"""
struct Interaction{F1, F2, F3, F4}
    energe::F1  # 势能函数
    force::F2   # 力函数
    cutenerge::F3  # 截断势能函数
    cutforce::F4   # 截断力函数
    cutoff::Float64  # 截断距离
    cutrg::Float64   # 截断范围


    # 定义构造函数
    function Interaction(energe::F1, force::F2, cutoff::Float64, cutrg::Float64) where {F1, F2}
        

        # 初始化截断势能和力的参数
        Er1 = 0
        dU1 = 0
        Er2 = energe(cutoff - cutrg)
        dU2 = (force(Vector([cutoff - cutrg,0,0])))[1]
        a = -(dU2 - dU1) / (2 * cutrg)
        b = -2 * a * (cutoff )
        c = Er1 - a * (cutoff)^2 - b * (cutoff)

        # 定义截断势能函数
        function cutenerge(r::Float64)
            nr = abs(r)
            if nr > cutoff
                return 0.0
            elseif nr < cutoff - cutrg
                return energe(r)
            else
                return a * nr^2 + b * nr + c
            end
        end

        # 定义截断力函数，接收向量输入
        function cutforce(r::Vector{Float64})
            nr = norm(r)
            if nr > cutoff
                return Vector(zeros(3))
            elseif nr < cutoff - cutrg
                return force(r)
            else
                return ((2 * a * nr + b) / nr) * r
            end
        end

        # 定义截断力函数的重载，使其可以接收 Float64 类型的输入
        cutforce(r::Float64) = (cutforce(Vector([r, 0, 0])))[1]

        # 返回新的 Interaction 实例
        new{F1, F2, typeof(cutenerge), typeof(cutforce)}(energe, force, cutenerge, cutforce, cutoff, cutrg)
    end
end


"""
计算晶胞中i,j原子之间的相互作用能
:param cell: 晶胞
:param interaction: 相互作用
:param i,j: 原子序号
:param ifnormize: 是否归一化，默认为 true,将用配位数对能量进行归一化
:param maxiter: 最大迭代次数，默认为 -1,将自动计算
:param tol: 误差，默认为 1e-3 用于判断div 0错误
"""
function cell_energeij(cell::UnitCell,interaction::Interaction,i::Int,j::Int;ifnormlize::Bool=true,maxiter::Int=-1,tol::Float64=1e-3)

    cutoff=interaction.cutoff
    atomi=cell.atoms[i]
    atomj=cell.atoms[j]
    a,b,c=cell.copy
    if maxiter==-1
        maxiter=Int(ceil(cutoff/minimum((cell.lattice_vectors*Vector([1,1,1])))))+1
    end
    cni=atomi.cn
    rij=atomj.position-atomi.position
    rij_lt=cell.lattice_vectors*rij

    energe=-interaction.cutenerge(norm(rij_lt))
    for ci in 0:maxiter
        for bi in 0:maxiter
            for ai in 0:maxiter
                rij=atomj.position-atomi.position+ai*Vector([a,0,0])+bi*Vector([0,b,0])+ci*Vector([0,0,c])
                rij_lt=cell.lattice_vectors*rij
                r=norm(rij_lt)
        
                if r>cutoff
                    break
                end
                if r<tol
                    continue
                end
                energe+=interaction.cutenerge(r)
            end
        
        end

    end
    for ci in 0:-1:-maxiter
        for bi in 0:-1:-maxiter
            for ai in 0:-1:-maxiter
                rij=atomj.position-atomi.position+ai*Vector([a,0,0])+bi*Vector([0,b,0])+ci*Vector([0,0,c])
                rij_lt=cell.lattice_vectors*rij
                r=norm(rij_lt)
                if r>cutoff
                    break
                end
                if r<tol
                    continue
                end
                energe+=interaction.cutenerge(r)
            end
        end

    end
    if ifnormlize
        energe=energe/cni
    end
    return energe
end

"""
计算晶胞的总能量
:param cell: 晶胞
:param interaction: 相互作用
:param ifnormize: 是否归一化，默认为 true,将用配位数对能量进行归一化
:param maxiter: 最大迭代次数，默认为 -1,将自动计算
:param tol: 误差，默认为 1e-3 用于判断div 0错误
"""
function cell_energe(cell::UnitCell,interaction::Interaction;ifnormlize::Bool=true,maxiter::Int=-1,tol::Float64=1e-3)
    energe=0.0
    for i in 1:length(cell.atoms)
        for j in 1:length(cell.atoms)
            if i!=j
                energe+=cell_energeij(cell,interaction,i,j,ifnormlize=ifnormlize,maxiter=maxiter,tol=tol)
            end
        end
    end
    return energe
end


end