"""
@author:XYC
@email:22307110070@m.fudan.edu.cn
The basic types and properties of unit cells are implemented and calculated here.
"""
module Model
    
using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using Base.Threads
export Atom, UnitCell, copycell, visualize_unitcell_atoms, Interaction, cell_energyij, cell_energy,cell_energyij0, cell_energy0, cell_forceij, cell_forcei, force_tensor,kb,visualize_unitcell_atoms0,filtercell,cell_forceij!,cell_forcei!,force_tensor!,cell_temp
    
global const kb=1.0

"""
原子类型
:param position: 原子位置,注意我们采用晶格坐标系,以便于对晶胞进行形变
:param momentum: 原子动量
:param mass: 原子质量
:param cn: 配位数
:param bound 边界状态 [1,1,0]表示处于x,y的0,0边界上 cn=2^(bound),[-1,-1,-1]表示位于a,b,c
"""
mutable struct Atom
    position::Vector{Float64}
    momentum::Vector{Float64}
    mass::Float64
    cn::Int
    bound::Vector{Int}
    function Atom(position::Vector{Float64},momentum::Vector{Float64},mass::Float64,cn::Int,bound::Vector{Int})
        new(position,momentum,mass,cn,bound)
    end
    function Atom(position::Vector{Float64})
        new(position,zeros(3),1.0,1,[0,0,0])
    end
    function Atom(position::Vector{Float64},momentum::Vector{Float64})
        new(position,momentum,1.0,1,[0,0,0])
    end

    function Atom(position::Vector{Float64},mass::Float64)
        new(position,zeros(3),mass,1,[0,0,0])
    end
    function Atom(position::Vector{Float64},momentum::Vector{Float64},mass::Float64)
        new(position,momentum,mass,1,[0,0,0])
    end
end



"""
晶胞类型
:param lattice_vectors: Matrix{Float64} 晶格矢量
:param atoms:Vector{Atom} 原子数组
:param copy:Vector{Int} 晶胞复制次数,default=[1,1,1],表示以原点为中心复制4个元胞,这是为了保证NPT系综元胞体积变化的各项同性
:param Volume: Float64 晶胞体积,default=det(lattice_vectors)*copy[1]*copy[2]*copy[3]*8 初始化时可自动生成
"""
mutable struct UnitCell
    lattice_vectors::Matrix{Float64}
    atoms::Vector{Atom}
    copy::Vector{Int}
    Volume::Float64
    function UnitCell(lattice_vectors::Matrix{Float64}, atoms::Vector{Atom}, copy::Vector{Int})
        # println(copy[1]*copy[2]*copy[3])
        # println(det(lattice_vectors))
        v=copy[1]*copy[2]*copy[3]*det(lattice_vectors)*8
        new(lattice_vectors, atoms,copy,v)
    end
    function UnitCell(lattice_vectors::Matrix{Float64}, atoms::Vector{Atom})
        v=det(lattice_vectors)*8
        new(lattice_vectors, atoms, Vector([1,1,1]),v)
    end
    function UnitCell(lattice_vectors::Adjoint{Float64, Matrix{Float64}}, atoms::Vector{Atom})
        v=det(lattice_vectors)*8
        new(lattice_vectors, atoms, Vector([1,1,1]),v)
    end
end


"""
复制晶胞
:param cell:UnitCell 晶胞
:param a,b,c:Int 复制次数,其将关于原点对称复制8*abc个,并赋予bound和cn
:param tol: 误差,default=0.001,用于计算配位数
"""
function copycell(cell::UnitCell,a::Int,b::Int,c::Int,tol::Float64=0.001)::UnitCell
    atoms=Vector{Atom}([])
    pl=Vector{Vector{Float64}}([])
    cp=[a,b,c]
    for atom in cell.atoms
        for i in -a:a-1
            for j in -b:b-1
                for k in -c:c-1
                    newp=atom.position+Vector([i,j,k])
                    pm=atom.momentum
                    m=atom.mass
                    if newp in pl
                        continue
                    end
                    
                    bd=Vector{Int}([0,0,0])
                    ct=0
                    for i in 1:3
                        bound1=cp[i]*1.0
                        bound2=cp[i]*(-1.0)
                        if abs(bound1-newp[i])<tol
                                bd[i]-=1
                                ct+=1
                        elseif abs(bound2-newp[i])<tol
                                bd[i]+=1
                                ct+=1
                        end
                    end
                    cn=2^ct
                    push!(atoms,Atom(newp,pm,m,cn,bd))
                    push!(pl,newp)
                end
            end
        end      
    end
    
    return UnitCell(cell.lattice_vectors,atoms,Vector([a,b,c]))
end

"""
过滤晶胞中的边界原子
:param cell: UnitCell晶胞
:return: 过滤后的晶胞
"""
function filtercell(cell::UnitCell)
    natom=length(cell.atoms)
    atoms=Vector{Atom}([])
    for i in 1:natom
        if -1 in cell.atoms[i].bound
            continue
        end
        push!(atoms,cell.atoms[i])
    end
    return UnitCell(cell.lattice_vectors,atoms,cell.copy)
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
相互作用类型,通过二次函数添加截断于cutoff-cutrg - cutoff
:param energe: 势能函数
:param force: 力函数
:param cutenerge: 截断势能函数
:param cutforce: 截断力函数
:param cutoff: 截断距离
:param cutrg: 截断范围
"""
struct Interaction{F1, F2, F3, F4}
    energy::F1  # 势能函数
    force::F2   # 力函数
    cutenergy::F3  # 截断势能函数
    cutforce::F4   # 截断力函数
    cutoff::Float64  # 截断距离
    cutrg::Float64   # 截断范围


    # 定义构造函数
    function Interaction(energy::F1, force::F2, cutoff::Float64, cutrg::Float64) where {F1, F2}
        

        # 初始化截断势能和力的参数
        Er2 = energy(cutoff - cutrg)
        dU2 = -(force(Vector([cutoff - cutrg,0,0])))[1]
        bb=Vector{Float64}([0.0,0.0,dU2,Er2])
        x1=cutoff
        x2=cutoff-cutrg
        A=[x1^3 x1^2 x1 1;3*x1^2 2*x1 1 0;3*x2^3 2*x2 1 0;x2^3 x2^2 x2 1]
        a,b,c,d=inv(A)*bb

        # 定义截断势能函数
        function cutenergy(r::Float64)
            nr = abs(r)
            if nr > cutoff
                return 0.0
            elseif nr < cutoff - cutrg
                return energy(r)
            else
                return a*nr^3+b*nr^2+c*nr+d
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
                return -((3 * a * nr^2 +2*b*nr+c) / nr) * r
            end
        end

        # 定义截断力函数的重载，使其可以接收 Float64 类型的输入
        cutforce(r::Float64) = (cutforce(Vector([r, 0, 0])))[1]

        # 返回新的 Interaction 实例
        new{F1, F2, typeof(cutenergy), typeof(cutforce)}(energy, force, cutenergy, cutforce, cutoff, cutrg)
    end
end


"""
计算晶胞中i,j原子之间的相互作用能,遍历周期,用于测试和cutoff>box/2的情况,且会自动忽略间距过小<tol的原子,通常使用cell_energyij0
之前调试时用的，现在基本弃用
:param cell: 晶胞
:param interaction: 相互作用
:param i,j: 原子序号
:param ifnormize: 是否归一化，默认为 true,将用配位数对能量进行归一化
:param maxiter: 最大迭代次数，默认为 -1,将自动计算
:param tol: 误差，默认为 1e-3 用于判断div 0错误
"""
function cell_energyij(cell::UnitCell, interaction::Interaction, i::Int, j::Int; ifnormalize::Bool=true, maxiter::Int=-1, tol::Float64=1e-3)
    cutoff = interaction.cutoff
    atomi = cell.atoms[i]
    atomj = cell.atoms[j]
    a, b, c = cell.copy*2
    if maxiter == -1
        maxiter = Int(ceil(cutoff / minimum(cell.lattice_vectors * @SVector [1, 1, 1]))) 
    end
    cni = atomi.cn
    base_rij = SVector{3}(atomj.position - atomi.position)
    # 预先计算这些向量
    a_vector = @SVector [a, 0, 0]
    b_vector = @SVector [0, b, 0]
    c_vector = @SVector [0, 0, c]
    energy = -interaction.cutenergy(norm(cell.lattice_vectors * base_rij))
    
    for ci in 0:maxiter
        for bi in 0:maxiter
            for ai in 0:maxiter
                temp_rij = base_rij + ai * a_vector + bi * b_vector + ci * c_vector
                temp_rij_lt = cell.lattice_vectors * temp_rij
                r = norm(temp_rij_lt)
                if r > cutoff
                    continue
                end
                if r < tol
                    continue
                end
                # println("ij cal $temp_rij at $ai,$bi,$ci")
                energy += interaction.cutenergy(r)
            end
        end
    end

    for ci in 0:-1:-maxiter
        for bi in 0:-1:-maxiter
            for ai in 0:-1:-maxiter
                temp_rij = base_rij + ai * a_vector + bi * b_vector + ci * c_vector
                temp_rij_lt = cell.lattice_vectors * temp_rij
                r = norm(temp_rij_lt)
                # println("ij cal $temp_rij at $ai,$bi,$ci,$r")
                if r > cutoff
                    # println("r>ct at $r $ai")
                    continue
                end
                if r < tol
                    continue
                end
                energy += interaction.cutenergy(r)
            end
        end
    end
    
    if ifnormalize
        energy /= cni
    end
    
    return energy
end

"""
计算晶胞的总能量,遍历周期,用于测试和cutoff>box/2的情况,通常使用cell_energy0,基本弃用
:param cell: 晶胞
:param interaction: 相互作用
:param ifnormize: 是否归一化，默认为 true,将用配位数对能量进行归一化
:param maxiter: 最大迭代次数，默认为 -1,将自动计算
:param tol: 误差，默认为 1e-3 用于判断div 0错误
"""
function cell_energy(cell::UnitCell,interaction::Interaction;ifnormalize::Bool=true,maxiter::Int=-1,tol::Float64=1e-3)
    a,b,c=cell.copy
    # if interaction.cutoff>minimum((cell.lattice_vectors*Vector([a,b,c])))
    #     lv=cell.lattice_vectors*Vector([a,b,c])
    #     cutoff=interaction.cutoff
    #     println("Warning: Cutoff $cutoff is larger than the minimum distance of the lattice vectors $lv ,energe i and Ri will be lost")
    # end
    energy=0.0
    for i in 1:length(cell.atoms)
        for j in 1:length(cell.atoms)
            if i!=j
                energy+=cell_energyij(cell,interaction,i,j,ifnormalize=ifnormalize,maxiter=maxiter,tol=tol)
                # println("energy at $i $j is $energy")
            end
        end
    end
    return energy/2
end

"""
计算cell温度,减去了质心动能
:param cell: 晶胞
"""
function cell_temp(cell::UnitCell)
    kb=1.0
    Ek=0.0
    for atom in cell.atoms
        p=atom.momentum
        Ek+=sum(p.^2)/(atom.mass)/2
    end
    return 2*Ek/(3*kb*(length(cell.atoms)-1))    
end

"""
采用邻近最小像计算cell能量,assert:cutoff<box/2,若晶胞原子有重复,iffilter=true将根据bound属性去重
:param cell: 晶胞
:param interaction: 相互作用
:param i,j: 原子序号、
:param ifnormize: 是否归一化，默认为 true,将用配位数对能量进行归一化
"""
function cell_energyij0(cell::UnitCell, interaction::Interaction, i::Int, j::Int; ifnormalize::Bool=false,iffilter::Bool=false)
    cutoff = interaction.cutoff
    atomi = cell.atoms[i]
    atomj = cell.atoms[j]
    bd=abs.(atomj.bound)
    cp = cell.copy
    cni = atomi.cn
    rij=atomj.position-atomi.position
    energy=0.0
    for k in 1:3
        rijk=rij[k]
        cpk=cp[k]
        if iffilter
            if bd[k]==1
                continue
            end
        end
        if rijk>cpk
                rij[k]=rijk-cpk*2.0
        elseif rijk<-cpk
                rij[k]=rijk+cpk*2.0
        end
    end
    # println("ij0 cal $rij")
    rij=cell.lattice_vectors*rij
    nr=norm(rij)
    if nr>cutoff
        energy=0.0
    else
        energy=interaction.cutenergy(nr)
        if ifnormalize
            energy /= cni
        end
    end
    return energy
end



"""
计算晶胞的总能量,采用邻近最小像,assert:cutoff<box/2
:param cell: 晶胞
:param interaction: 相互作用
:param ifnormize: 是否归一化，默认为 true,将用配位数对能量进行归一化
:param maxiter: 最大迭代次数，默认为 -1,将自动计算
:param tol: 误差，默认为 1e-3 用于判断div 0错误
"""
function cell_energy0(cell::UnitCell,interaction::Interaction;ifnormalize::Bool=false,iffilter::Bool=false)
    # a,b,c=cell.copy
    # if interaction.cutoff>(maximum(cell.lattice_vectors)*minimum(Vector([a,b,c])))
    #     println(a,b,c)
    #     lv=cell.lattice_vectors*Vector([a,b,c])
    #     cutoff=interaction.cutoff
    #     println("Warning: Cutoff $cutoff is larger than the minimum distance of the lattice vectors $lv ,energy i and Ri will be lost under rules of nearest neighbor image")
    # end
    energy=0.0
    for i in 1:length(cell.atoms)
        for j in 1:length(cell.atoms)
            if i!=j
                energy+=cell_energyij0(cell,interaction,i,j,ifnormalize=ifnormalize,iffilter=iffilter)
                # println("energy at $i $j is $energy")
            end
        end
    end
    return energy/2
end

"""
计算晶胞中i,j原子之间的相互作用力,采用邻近最小像,assert:cutoff<box/2,若晶胞原子有重复,iffilter=true将根据bound属性去重
:param cell: 晶胞
:param interaction: 相互作用
:param i,j: 原子序号
这里rij其实应该是rji,一开始搞错了,故最后加负号
"""
function cell_forceij(cell::UnitCell, interaction::Interaction, i::Int, j::Int;iffilter::Bool=false)
    cutoff = interaction.cutoff
    atomi = cell.atoms[i]
    atomj = cell.atoms[j]
    bd=abs.(atomj.bound)
    cp = cell.copy
    cni = atomi.cn
    rij=atomj.position-atomi.position
    for k in 1:3
        rijk=rij[k]
        cpk=cp[k]
        if iffilter
            if bd[k]==1
                continue
            end
        end
        if rijk>cpk
                rij[k]=rijk-cpk*2
        elseif rijk<-cpk
                rij[k]=rijk+cpk*2
        end
    end
    rij=cell.lattice_vectors*rij
    nr=norm(rij)
    if nr>cutoff
        force=zeros(3)
    else
        force=interaction.cutforce(rij)
    end

    if any(isnan,force)
        throw("Warning the same atoms of atom i=$i j=$j lead to the nan force")
    end
    return -force
end

"""
计算晶胞中i原子的受力
:param cell: 晶胞
:param interaction: 相互作用
:param i: 原子序号
"""
function cell_forcei(cell::UnitCell,interaction::Interaction,i::Int;iffilter::Bool=false)
    forcei=zeros(3)
        for j in 1:length(cell.atoms)
            if i!=j
                forcei+=cell_forceij(cell,interaction,i,j,iffilter=iffilter)
            end
        end
    return forcei
end


"""
计算晶胞的应力张量,待加入dU/dh_i的项
:param cell: 晶胞
:param interaction: 相互作用
"""
function force_tensor(cell::UnitCell,interaction::Interaction)
    tensor=zeros(3,3)
    v=cell.Volume
    for i in 1:length(cell.atoms)
        for a in 1:3
            for b in 1:3
                forcei=cell_forcei(cell,interaction,i)
                atom=cell.atoms[i]
                tensor[a,b]+=forcei[a]*atom.position[b]/2+atom.momentum[a]*atom.momentum[b]/atom.mass
            end
        end
    end
    return tensor./v
end


"""
计算晶胞中i,j原子之间的相互作用力,遍历计算,仅调试用
:param cell: 晶胞
:param interaction: 相互作用
:param i,j: 原子序号
:param ifnormize: 是否归一化，默认为 true,将用配位数对能量进行归一化
:param maxiter: 最大迭代次数，默认为 -1,将自动计算
:param tol: 误差，默认为 1e-3 用于判断div 0错误
"""
function cell_forceij!(cell::UnitCell, interaction::Interaction, i::Int, j::Int;maxiter::Int=-1)
    cutoff = interaction.cutoff
    atomi = cell.atoms[i]
    atomj = cell.atoms[j]
    a, b, c = cell.copy*2
    if maxiter == -1
        maxiter = Int(ceil(cutoff / minimum(cell.lattice_vectors * @SVector [1, 1, 1]))) 
    end
    base_rij = SVector{3}(atomj.position - atomi.position)
    # 预先计算这些向量
    a_vector = @SVector [a, 0, 0]
    b_vector = @SVector [0, b, 0]
    c_vector = @SVector [0, 0, c]
    force = -interaction.cutforce(cell.lattice_vectors * base_rij)
    for ci in 0:maxiter
        for bi in 0:maxiter
            for ai in 0:maxiter
                temp_rij = base_rij + ai * a_vector + bi * b_vector + ci * c_vector
                temp_rij_lt = cell.lattice_vectors * temp_rij
                r = norm(temp_rij_lt)
                if r > cutoff
                    continue
                end
                force += interaction.cutforce(temp_rij_lt)
            end
        end
    end

    for ci in 0:-1:-maxiter
        for bi in 0:-1:-maxiter
            for ai in 0:-1:-maxiter
                temp_rij = base_rij + ai * a_vector + bi * b_vector + ci * c_vector
                temp_rij_lt = cell.lattice_vectors * temp_rij
                r = norm(temp_rij_lt)
                if r > cutoff
                    continue
                end
                force+= interaction.force(temp_rij_lt)
            end
        end
    end
    
    return -force
end

"""
计算晶胞中i原子的受力,遍历计算,仅调试用
"""
function cell_forcei!(cell::UnitCell,interaction::Interaction,i::Int;maxiter::Int=-1)
    force=zeros(3)

        for j in 1:length(cell.atoms)
            if i!=j
                force+=cell_forceij!(cell,interaction,i,j,maxiter=maxiter)
                # println("energy at $i $j is $energy")
            end
        end
 
    return force
end

"""
计算晶胞的应力张量,遍历周期,仅调试用
:param cell: 晶胞
:param interaction: 相互作用
"""
function force_tensor!(cell::UnitCell,interaction::Interaction)
    tensor=zeros(3,3)
    v=cell.Volume
    for a in 1:3
        for b in 1:3
            for i in 1:length(cell.atoms)
                forcei=cell_forcei!(cell,interaction,i)
                atom=cell.atoms[i]
                tensor[a,b]+=forcei[a]*atom.position[b]/2+atom.momentum[a]*atom.momentum[b]/atom.mass
            end
        end
    end
    return tensor./v
end
end