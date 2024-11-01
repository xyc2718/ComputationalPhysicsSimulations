
module cellmin

using StaticArrays
# using Plots
using LinearAlgebra
# using Makie
using GLMakie 
using LsqFit
using ..Model

    
export birch_murnaghan,BMfit,gradientStep!,minimizeEnergy!,gradientDescent!

"""
BM 函数
:param V: 体积
:param p: 参数=[V0, B0, B0', E0]
"""
function birch_murnaghan(V::Vector{Float64}, p::Vector{Float64})
    V0, B0, B0_prime, E0 = p
    eta = (V0 ./ V).^(2/3)  # 使用广播运算符 ./
    E = E0 .+ (9 * V0 * B0 / 16) .* ((eta .- 1).^3 .* B0_prime .+ (eta .- 1).^2 .* (6 .- 4 .* eta))
    return E
end

function birch_murnaghan(V::Float64, p::Vector{Float64})
    V0, B0, B0_prime, E0 = p
    eta = (V0 ./ V).^(2/3)  # 使用广播运算符 ./
    E = E0 .+ (9 * V0 * B0 / 16) .* ((eta .- 1).^3 .* B0_prime .+ (eta .- 1).^2 .* (6 .- 4 .* eta))
    return E
end


"""
使用BM 拟合 E-V曲线
:param cl: cell
:param el: energy list
:param p0: 初始参数=[V0, B0, B0', E0]
"""
function BMfit(vl::Vector{Float64},el::Vector{Float64},cpcell::UnitCell,p0::Vector{Float64}=[1200.0, -10.0, 4.0, -1200.0])
    fit = curve_fit(birch_murnaghan, vl, el, p0)
    # 拟合结果
    fitted_params = fit.param
    min_lattice_constant = (fitted_params[1]/cpcell.copy[1]/cpcell.copy[2]/cpcell.copy[3])^(1/3)
    println("Fitted parameters:")
    println("V0 = ", fitted_params[1])
    println("B0 = ", fitted_params[2])
    println("B0' = ", fitted_params[3])
    println("E0 = ", fitted_params[4])
    println("latticeconstant = ",min_lattice_constant )
    fig = Figure()
    fitfunc(x)=birch_murnaghan(x, fitted_params)
    # 创建一个 Axis，并设置坐标范围
    ax = Axis(fig[1, 1],xlabel="Volume", ylabel="Energy", title="Birch-Murnaghan EOS Fit"
              )
    x=LinRange(minimum(vl),maximum(vl),1000)
    scatter!(ax,vl,el,color=:red,label="Data")
    lines!(ax,x,fitfunc.(x),color=:blue,label="Fit")
    axislegend(ax, position=:rt)
    fig
    return min_lattice_constant,fig
    end

    function gradientDescent!(cell::UnitCell,interaction::Interaction;ap::Float64=0.01,tol=1e-8,maxiter=1000,checktime=10)
        El=Vector{Float64}([])
        converge=false
        for s in 1:maxiter
           
            gradientStep!(cell,interaction,ap=ap)
            if mod(s,checktime)==0
                Ei=cell_energy(cell,interaction)
                println("$step=$s,E=$Ei")
                push!(El,Ei)
                if s>checktime
                    if abs(El[end]-El[end-1])<tol
                        converge=true
                        println("Energy is converge within $s steps of tol=$(tol) !")
                        break
                    end
                end
            end
        end
        if !converge
            println("Warning!Energy is not converge within $maxiter steps of tol=$tol !")
        end
        return El
    end

    function gradientStep!(cell::UnitCell,interaction::Interaction;ap::Float64=0.01)
        ltv=cell.lattice_vectors
        invlt=inv(ltv)
        m=cell.atoms[1].mass
        cp=cell.copy
        #  println(fl)
        #  println(cell_forcei(cell,interaction,32))
         for i in eachindex(cell.atoms)
            fi=cell_forcei(cell,interaction,i)
            cell.atoms[i].position+=ap*invlt*fi
            # println(ap*invlt*fl[i])
            for k in 1:3
             cell.atoms[i].position[k]=mod(cell.atoms[i].position[k]+cp[k],2*cp[k])-cp[k]
             end
         end
         update_rmat!(cell) 
         update_fmat!(cell,interaction)
    end


    function minimizeEnergy!(cell::UnitCell,interaction::Interaction;rg::Vector{Float64}=[1.0,10.0],n::Int=1000)
        El=Vector{Float64}([])
        cl = range(rg[1], stop=rg[2], length=n)
        for lt in cl
            cell.lattice_vectors=collect((Matrix([
                lt 0.0 0.0; #a1
                0.0 lt 0.0; #a2
                0.0 0.0 lt] #a3
            ))')
            update_rmat!(cell)
            # update_fmat!(cell,interaction)
            Ei=cell_energy(cell,interaction)
            push!(El,Ei)
        end
        minindex=argmin(El)
        lt=cl[minindex]
        cell.lattice_vectors=collect((Matrix([
            lt 0.0 0.0; #a1
            0.0 lt 0.0; #a2
            0.0 0.0 lt] #a3
        ))')
        cp=cell.copy
        cell.Volume=det(cell.lattice_vectors)*8*cp[1]*cp[2]*cp[3]
        update_rmat!(cell)
        update_fmat!(cell,interaction)
        return cl,El
    end

    
    function minimizeEnergyWithGradient!(cell::UnitCell,interaction::Interaction;rg::Vector{Float64}=[1.0,10.0],n::Int=1000,maxiter::Int=100,tol::Float64
        =1e-6)
        El=Vector{Float64}([])
        cl = range(rg[1], stop=rg[2], length=n)
        for lt in cl
            cell.lattice_vectors=collect((Matrix([
                lt 0.0 0.0; #a1
                0.0 lt 0.0; #a2
                0.0 0.0 lt] #a3
            ))')
            gradientDescent!(cell,interaction,maxiter=maxiter,tol=tol)
            Ei=cell_energy(cell,interaction)
            push!(El,Ei)
        end
        minindex=argmin(El)
        lt=cl[minindex]
        cell.lattice_vectors=collect((Matrix([
            lt 0.0 0.0; #a1
            0.0 lt 0.0; #a2
            0.0 0.0 lt] #a3
        ))')
        gradientDescent!(cell,interaction,maxiter=maxiter,tol=tol)
        return cl,El
    end

    function MonteCarlo!(cell::UnitCell,interaction::Interaction,ap=0.001,maxstep=100)
        E=cell_energy(cell,interaction)
        for i in maxstep
            dcell=deepcopy(cell)
            natom=length(cell.atoms)
            id=rand(1:natom)
            dcell[id].position+=ap*randn(3)
            update_rmat!(dcell)
            dE=cell_energy(dcell,interaction)
            if dE<E
                cell=dcell
                E=dE
            end
        end

    end
end