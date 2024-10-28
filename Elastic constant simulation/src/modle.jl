"""
@author:XYC
@email:22307110070@m.fudan.edu.cn
The basic types and properties of unit cells are implemented and calculated here.
"""
module Model
    
using StaticArrays
using LinearAlgebra
using Base.Threads
using IterTools
export Atom, UnitCell, copycell,  Interaction, cell_energyij, cell_energy,cell_energyij0, cell_energy0, cell_forceij, cell_forcei, force_tensor,filtercell,cell_forceij!,cell_forcei!,force_tensor!,cell_temp,Ngradient0,getrij,is_diagonal_matrix,Embedding,dUdhij,randcell!,Force_Tensor,getrij0,update_rmat!,update_rmati!,update_fmat!,cell_forcei0,set_lattice_vector!,apply_PBC!
    
global const kb=8.617332385e-5 #eV/K

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
    rmat::Matrix{SVector{3,Float64}}
    ifrmat::Bool
    fmat::Vector{SVector{3,Float64}}
    iffmat::Bool
    function UnitCell(lattice_vectors::Matrix{Float64}, atoms::Vector{Atom}, copy::Vector{Int})
        # println(copy[1]*copy[2]*copy[3])
        # println(det(lattice_vectors))
        v=copy[1]*copy[2]*copy[3]*det(lattice_vectors)*8
        new(lattice_vectors, atoms,copy,v,Matrix{SVector{3,Float64}}(undef,length(atoms),length(atoms)),false,Vector{SVector{3,Float64}}(undef,length(atoms)),false)
    end
    function UnitCell(lattice_vectors::Matrix{Float64}, atoms::Vector{Atom})
        v=det(lattice_vectors)*8
        new(lattice_vectors, atoms, Vector([1,1,1]),v,Matrix{SVector{3,Float64}}(undef,length(atoms),length(atoms)),false,Vector{SVector{3,Float64}}(undef,length(atoms)),false)
    end
    function UnitCell(lattice_vectors::Adjoint{Float64, Matrix{Float64}}, atoms::Vector{Atom})
        v=det(lattice_vectors)*8
        new(lattice_vectors, atoms, Vector([1,1,1]),v,Matrix{SVector{3,Float64}}(undef,length(atoms),length(atoms)),false,Vector{SVector{3,Float64}}(undef,length(atoms)),false)
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






"""
通过最小近邻像获取ij间真实距离
若晶格向量不正交则会遍历周围26个晶胞
cell::UnitCell
i::Int
j::Int

return rij
"""
function getrij0(cell::UnitCell, i::Int, j::Int; diagonal_mat::Bool=true) :: SVector{3, Float64}
    if i==j
        return SVector{3}(0.0, 0.0, 0.0)
    end

    cp = SVector{3}(cell.copy)  # 静态数组
    ltv = SMatrix{3,3}(cell.lattice_vectors)  # 静态矩阵
    rij = SVector{3}(cell.atoms[j].position - cell.atoms[i].position)  # 3D 向量

    if diagonal_mat
        is_diagonal = is_diagonal_matrix(ltv)  # 使用 isdiagonal 函数
    else
        is_diagonal = true
    end

    if is_diagonal
        for k in 1:3
            rijk = rij[k]
            cpk = cp[k]
            if rijk > cpk
                rij = setindex(rij, rijk - cpk * 2.0, k)
            elseif rijk < -cpk
                rij = setindex(rij, rijk + cpk * 2.0, k)
            end
        end
        return ltv * rij  # 直接返回
    else
        values = [1.0, 0.0, -1.0]
        cp = cp .* 2.0  # 提前计算

        # 准备遍历组合
        combinations = Iterators.product(values .* cp[1], values .* cp[2], values .* cp[3])

        min_rij = rij
        min_norm2 = norm(ltv * rij)^2  # 使用平方模长

        for combo in combinations
            if combo != (0.0, 0.0, 0.0)
                new_rij = rij .+ SVector(combo...)
                norm2_value = norm(ltv * new_rij)^2  # 使用平方模长避免 sqrt
                if norm2_value < min_norm2
                    min_norm2 = norm2_value
                    min_rij = new_rij
                end
            end
        end

        return ltv * min_rij
    end
end

function update_rmati!(cell::UnitCell,i::Int)
    for j in 1:length(cell.atoms)
        rij=getrij0(cell,i,j)
        cell.rmat[i,j] = rij
        cell.rmat[j,i] = -rij
    end
end

function update_rmat!(cell::UnitCell)
    cell.ifrmat = true
    atom=length(cell.atoms)
Threads.@threads for i in 1:atom
        for j in i:atom
            rij=getrij0(cell,i,j)
            cell.rmat[i,j] = rij
            cell.rmat[j,i] = -rij
        end
    end
end





function getrij(cell::UnitCell, i::Int, j::Int) :: SVector{3, Float64}
    if  cell.ifrmat
        return cell.rmat[i,j]
    else
        return getrij0(cell,i,j)
    end
end






"""
判断是否为对角阵
"""
function is_diagonal_matrix(A::Matrix{Float64})
    rows, cols = size(A)
    if rows != cols
        return false
    end
    for i in 1:rows
        for j in 1:cols
            if i != j && A[i, j] != 0
                return false
            end
        end
    end
    
    return true
end

"""
判断是否为对角阵
"""
function is_diagonal_matrix(A::SMatrix{Float64})
    rows, cols = size(A)
    if rows != cols
        return false
    end
    for i in 1:rows
        for j in 1:cols
            if i != j && A[i, j] != 0
                return false
            end
        end
    end
    
    return true
end


function is_diagonal_matrix(mat::SMatrix{N, N, T, L}) where {N, T, L}
    for i in 1:N
        for j in 1:N
            if i != j && mat[i, j] != 0
                return false
            end
        end
    end
    return true
end


function Default_Embedding_energy(cell::UnitCell)
    return 0.0
    end
function Default_Embedding_force(cell::UnitCell,i::Int)
    return SVector{3}(0.0,0.0,0.0)
end
"""
嵌入能项
"""
struct Embedding{E1, E2}
    embedding_energy::E1
    embedding_force::E2
    ifembedding::Bool
    function Embedding() 
        new{typeof(Default_Embedding_energy),typeof(Default_Embedding_force)}(Default_Embedding_energy,Default_Embedding_force, false)
    end
    function Embedding(embedding_energy::E1, embedding_force::E2) where {E1, E2}
        new{E1,E2}(embedding_energy, embedding_force, true)
    end
end




"""
相互作用类型,通过二次函数添加截断于cutoff-cutrg - cutoff
:param energe: 势能函数
:param force: 力函数 ::SVector{3, Float64} -> SVector{3, Float64}
:param cutenerge: 截断势能函数
:param cutforce: 截断力函数
:param cutoff: 截断距离
:param cutrg: 截断范围
:param embedding
"""
struct Interaction{F1, F2, F3, F4}
    energy::F1  # 势能函数
    force::F2   # 力函数
    cutenergy::F3  # 截断势能函数
    cutforce::F4   # 截断力函数
    cutoff::Float64  # 截断距离
    cutrg::Float64   # 截断范围
    embedding::Embedding


    # 定义构造函数
    function Interaction(energy::F1, force::F2, cutoff::Float64, cutrg::Float64) where {F1, F2}
        
        # 初始化截断势能和力的参数
        Er2 = energy(cutoff - cutrg)
        dU2 = -(force(SVector{3}(cutoff - cutrg,0,0)))[1]
        bb=Vector{Float64}([0.0,0.0,dU2,Er2])
        x1=cutoff
        x2=cutoff-cutrg
        A=[x1^3 x1^2 x1 1;3*x1^2 2*x1 1 0;3*x2^2 2*x2 1 0;x2^3 x2^2 x2 1]
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
        function cutforce(r::SVector{3, Float64})
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
        cutforce(r::Float64) = (cutforce(SVector{3}([r, 0, 0])))[1]

        # 返回新的 Interaction 实例
        new{F1, F2, typeof(cutenergy), typeof(cutforce)}(energy, force, cutenergy, cutforce, cutoff, cutrg,Embedding())
    end


     # 定义构造函数
     function Interaction(energy::F1, force::F2, cutoff::Float64, cutrg::Float64,embedding::Embedding) where {F1, F2}
        

        # 初始化截断势能和力的参数
        Er2 = energy(cutoff - cutrg)
        dU2 = -(force(SVector{3}(cutoff - cutrg,0,0)))[1]
        bb=Vector{Float64}([0.0,0.0,dU2,Er2])
        x1=cutoff
        x2=cutoff-cutrg
        A=[x1^3 x1^2 x1 1;3*x1^2 2*x1 1 0;3*x2^2 2*x2 1 0;x2^3 x2^2 x2 1]
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
        function cutforce(r::SVector{3, Float64})
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
        cutforce(r::Float64) = (cutforce(SVector{3}([r, 0, 0])))[1]

        # 返回新的 Interaction 实例
        new{F1, F2, typeof(cutenergy), typeof(cutforce)}(energy, force, cutenergy, cutforce, cutoff, cutrg,embedding)
    end
end





function update_fmat!(cell::UnitCell,interaction::Interaction)
    cell.iffmat = true
   atom=length(cell.atoms)
    Threads.@threads  for i in 1:atom
                fi=cell_forcei0(cell,interaction,i)
                cell.fmat[i] =fi
    end

end



"""
计算cell温度,减去了质心动能
:param cell: 晶胞
"""
function cell_temp(cell::UnitCell)
    kb=8.617332385e-5 #eV/K
    Ek=0.0
    for atom in cell.atoms
        p=atom.momentum
        Ek+=sum(p.^2)/(atom.mass)/2
    end
    return 2*Ek/(3*kb*(length(cell.atoms)-1))    
end

"""
采用邻近最小像计算cell能量,assert:cutoff<box/2,若晶胞原子有重复
:param cell: 晶胞
:param interaction: 相互作用
:param i,j: 原子序号、
:param ifnormize: 是否归一化，默认为 true,将用配位数对能量进行归一化
"""
function cell_energyij(cell::UnitCell, interaction::Interaction, i::Int, j::Int; ifnormalize::Bool=false)
    cutoff = interaction.cutoff
    cni = cell.atoms[i].cn
    energy=0.0
    rij=getrij(cell,i,j)
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
function cell_energy(cell::UnitCell,interaction::Interaction;ifnormalize::Bool=false)
    # a,b,c=cell.copy
    # if interaction.cutoff>(maximum(cell.lattice_vectors)*minimum(Vector([a,b,c])))
    #     println(a,b,c)
    #     lv=cell.lattice_vectors*Vector([a,b,c])
    #     cutoff=interaction.cutoff
    #     println("Warning: Cutoff $cutoff is larger than the minimum distance of the lattice vectors $lv ,energy i and Ri will be lost under rules of nearest neighbor image")
    # end
    energy=0.0
    energyeb=0.0
    for i in 1:length(cell.atoms)
        for j in i+1:length(cell.atoms)
                energy+=cell_energyij(cell,interaction,i,j,ifnormalize=ifnormalize)
                # println("energy at $i $j is $energy")
        end
    end
    if interaction.embedding.ifembedding
        energyeb=interaction.embedding.embedding_energy(cell)
    end
    return energy+energyeb
end






"""
计算晶胞中i受到原子j的相互作用力,采用邻近最小像,assert:cutoff<box/2
:param cell: 晶胞
:param interaction: 相互作用
:param i,j: 原子序号
这里rij其实应该是rji,一开始搞错了,故最后加负号
"""
function cell_forceij(cell::UnitCell, interaction::Interaction, i::Int, j::Int)::SVector{3, Float64}
    cutoff = interaction.cutoff
    rij=getrij(cell,i,j)
    nr=norm(rij)
    if nr>cutoff
        force=SVector{3,Float64}(0.0, 0.0, 0.0)
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
function cell_forcei0(cell::UnitCell,interaction::Interaction,i::Int)::SVector{3,Float64}
    forcei=SVector{3}(0.0, 0.0, 0.0)
        for j in 1:length(cell.atoms)
            if i!=j
                forcei+=cell_forceij(cell,interaction,i,j)
            end
        end
    if interaction.embedding.ifembedding
        # forcei.-=Ngradient0(cell,i,interaction.embedding.embedding_energy,[])
        forcei+=interaction.embedding.embedding_force(cell,i)
    end
    return forcei
end

function cell_forcei(cell::UnitCell,interaction::Interaction,i::Int)\
    if cell.iffmat
        return cell.fmat[i]
    else
        return cell_forcei0(cell,interaction,i)
    end
end


function  Ngradient0(cell::UnitCell,i::Int,f::Function,para::Vector;dr::Vector{Float64}=[0.001,0.001,0.001])
    df=zeros(3)
    lt=cell.lattice_vectors
    invlt=inv(lt)
    drm=Diagonal(dr)
    for j in 1:3
        dri=invlt*drm[j,:]
        dcell=deepcopy(cell)
        dcell.atoms[i].position.+=dri
        update_rmati!(dcell,i)
        update_fmat!(dcell,interaction)
        f1=f(dcell,para...)
        dcell.atoms[i].position.-=2*dri
        update_rmati!(dcell,i)
        update_fmat!(dcell,interaction)
        f2=f(dcell,para...)
        df[j]=(f1-f2)/2/dr[j]
    end
    return df
end





"""
计算晶胞的应力张量,不含dU/dhij项，请使用dUdhij函数额外计算
:param cell: 晶胞
:param interaction: 相互作用
"""
function force_tensor(cell::UnitCell,interaction::Interaction)
    tensor=SMatrix{3, 3}(0, 0, 0, 0, 0, 0, 0, 0, 0)
    v=cell.Volume
    ltv=cell.lattice_vectors
    for i in 1:length(cell.atoms)
            forcei=cell_forcei(cell,interaction,i)
            ri=ltv*(cell.atoms[i].position)
            p=cell.atoms[i].momentum
            m=cell.atoms[i].mass
            tensor+=forcei*ri'+p*p'./m
    end
    return tensor./v
end


function randcell!(cell::UnitCell,k::Float64=0.1)
    for atom in cell.atoms
        atom.position+=k*randn(3)
    end
    update_rmat!(cell)
    update_fmat!(cell,interaction)
end

function Force_Tensor(cell::UnitCell,interaction::Interaction;dr::BigFloat=BigFloat("1e-8"))
    ft=force_tensor(cell,interaction)
    dUdh=dUdhij(cell,interaction,BigFloat("1e-8"))
    ltv=cell.lattice_vectors
    V=cell.Volume
    ft0=ft-(dUdh*transpose(ltv))./V
    return ft0
end

"""
差分计算dU/dhij
"""
function dUdhij(fcell::UnitCell,interaction::Interaction,dr::BigFloat=BigFloat("1e-8"))
    ltv=fcell.lattice_vectors
    dltv=dr*maximum(ltv)
    # println(dltv)
    re=zeros(3,3)
    cp=fcell.copy
    for i in 1:3
        for j in 1:3
            dcell=deepcopy(fcell)
            ltv2=deepcopy(ltv)
            ltv2[i,j]+=dltv
            for atom in dcell.atoms
                ri=(inv(ltv2))*ltv*atom.position
                    for k in 1:3
                        atom.position[k]=mod(ri[k]+cp[k],2*cp[k])-cp[k]
                    end
            end
            dcell.lattice_vectors=ltv2

            update_rmat!(dcell)
            update_fmat!(dcell,interaction)
            energy1=cell_energy(dcell,interaction)

            dcell=deepcopy(fcell)
            ltv2=deepcopy(ltv)
            ltv2[i,j]-=dltv
            # println(ltv2)
            for atom in dcell.atoms
                ri=(inv(ltv2))*ltv*atom.position
                    for k in 1:3
                        atom.position[k]=mod(ri[k]+cp[k],2*cp[k])-cp[k]
                    end
            end
            dcell.lattice_vectors=ltv2

            update_rmat!(dcell)
            update_fmat!(dcell,interaction)
            energy2=cell_energy(dcell,interaction)
            # println("i=$i,j=$j,dE=$(energy1-energy2)")
            re[i,j]=(energy1-energy2)/dltv/2
        end
    end

    return re
end


function apply_PBC!(cell::UnitCell,interaction::Interaction)
    a,b,c=cell.copy
    for i in eachindex(cell.atoms)
        for k in 1:3
            ri=cell.atoms[i].position
            ri[1]=mod(ri[1]+a,2*a)-a
            ri[2]=mod(ri[2]+b,2*b)-b
            ri[3]=mod(ri[3]+c,2*c)-c
            cell.atoms[i].position=ri
        end
    
    end
    update_rmat!(cell)
    update_fmat!(cell,interaction)

end

function set_lattice_vector!(cell::UnitCell,lt::Matrix{Float64},interaction::Interaction)
    lt0=cell.lattice_vectors
    cell.lattice_vectors=lt
    invlt=inv(lt)
    for i in eachindex(cell.atoms)
        cell.atoms[i].position=invlt*lt0*cell.atoms[i].position
    end
    apply_PBC!(cell,interaction)
end



####################################################################################
#以下为几乎弃用代码


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
function cell_energyij0(cell::UnitCell, interaction::Interaction, i::Int, j::Int; ifnormalize::Bool=true, maxiter::Int=-1, tol::Float64=1e-3)
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
function cell_energy0(cell::UnitCell,interaction::Interaction;ifnormalize::Bool=true,maxiter::Int=-1,tol::Float64=1e-3)
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
                energy+=cell_energyij0(cell,interaction,i,j,ifnormalize=ifnormalize,maxiter=maxiter,tol=tol)
                # println("energy at $i $j is $energy")
            end
        end
    end
    return energy/2
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