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
using Distributions
using JLD2

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

# 创建铜的原子列表
atoms = [Atom(pos) for pos in atom_positions]
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






function Nhcpisoint!(cell::UnitCell,interaction::Interaction,thermostatchain::Vector{Thermostat},barostat::Barostat,dt::Float64)
    kb=1.0
    natom=length(cell.atoms)
    Nf=3*natom-3
    T=thermostatchain[1].T
    GN1KT=(Nf)*kb*T
    GNKT=Nf*kb*T
    GKT=kb*T
    odnf=1+3/Nf
    W=barostat.W
    V=cell.Volume
    Pe=barostat.Pe
    Pint=pressure_int(cell,interaction)
    nnos=length(thermostatchain) ##恒温器数量3个
    glogs=zeros(nnos)
    vlogs=[th.Pt for th in thermostatchain]
    xlogs=[th.Rt for th in thermostatchain]
    xlogv=1/3*log(barostat.V)
    vlogv=barostat.Pv
    glogv=0.0

    
    nresn=3  #3*3差分 nresn->nc，第一次多步
    nyosh=3           #nyosh->nys 第二维多步
    w1=1/(2-2^(1/3))
    w3=w1
    w2=1-2*w1
    wdti=[w1,w2,w3]*dt
    wdti2=wdti./2/nresn
    wdti4=wdti./4/nresn
    wdti8=wdti./8/nresn

    
    scale::BigFloat=1.0
    kint=0.0
    for i in 1:natom 
        kint=kint+dot(cell.atoms[i].momentum,cell.atoms[i].momentum)/cell.atoms[i].mass
    end
    # println("kint=$kint")
        glogs[1]=(kint+W*vlogv^2-GN1KT)/thermostatchain[1].Q
        # println("glogs=$glogs")
        glogv=(odnf*kint+3.0*(Pint-Pe)*V)/W
        # println(W*vlogv^2)
        # println(vlogv^2)
        # println(GN1KT)
        # println(thermostatchain[1].Q)
    for iresn in 1:nresn 
        for iyosh in 1 :nyosh
                   
            vlogs[nnos]=vlogs[nnos]+glogs[nnos]*wdti4[iyosh]
            
            for inos in 1:nnos-1
                AA=exp(-wdti8[iyosh]*vlogs[nnos+1-inos])
                vlogs[nnos-inos]=vlogs[nnos-inos]*AA^2+wdti4[iyosh]*glogs[nnos-inos]*AA
            end
            AA=exp(-wdti8[iyosh]*vlogs[1])
            # println(vlogs)

            

            vlogv=vlogv*AA^2+wdti4[iyosh]*glogv*AA

            AA=exp(-wdti2[iyosh]*(vlogs[1]+odnf*vlogv))
            # println("scale=$scale,AA=$AA,vlogs=$vlogs")
            scale=scale*AA
            # println("scale=$scale,AA=$AA,vlogs[1]=$(vlogs[1]),vlogv=$vlogv")
            kint=kint*AA^2
            glogv=(odnf*kint+3.0*(Pint-Pe)*V)/W
            
    
            for inos in 1:nnos
                xlogs[inos]=xlogs[inos]+vlogs[inos]*wdti2[iyosh]
            end
            AA=exp(-wdti8[iyosh]*vlogs[1])
            vlogv=vlogv*AA^2+wdti4[iyosh]*glogv*AA
            glogs[1]=(kint+W*vlogv^2-GN1KT)/thermostatchain[1].Q
            
            for inos in 1: nnos-1
                AA=exp(-wdti8[iyosh]*vlogs[inos+1])
                vlogs[inos]=vlogs[inos]*AA*AA+wdti4[iyosh]*glogs[inos]*AA
                glogs[inos+1]=(thermostatchain[inos].Q*vlogs[inos]*vlogs[inos]-GKT)/thermostatchain[inos+1].Q
            end
            vlogs[nnos]=vlogs[nnos]+glogs[nnos]*wdti4[iyosh]
        end

    end

   
    for i in 1:natom 
        cell.atoms[i].momentum=cell.atoms[i].momentum*scale
    end

    # barostat.V=exp(3*xlogv)
    # cell.Volume=barostat.V
    # cell.lattice_vectors=cell.lattice_vectors*(barostat.V/V0)^(1/3)
    # println("V0=$V0,V=$V")
    barostat.Pv=vlogv
    for inos in 1:nnos 
        thermostatchain[inos].Rt=xlogs[inos]
        thermostatchain[inos].Pt=vlogs[inos]
    end
    # println(exp(3*xlogv))
    # println(xlogs)
    # println(vlogs)  
end


open("data_pthv3_NF-3.txt", "w") do io
    jldopen("data_pthv3_cell_NF-3.txt","w") do iojl
Ts=1.0
Ps=100.0
dt=0.001
dt2=dt/2


inicell=initcell(Ps,Ts,atoms,interaction,cp=[3,3,3],Prg=[1.0,6.0])
cell=deepcopy(inicell)
natom=length(inicell.atoms)
Qs=3*natom*Ts*(10*dt)^2
Ws=3*natom*Ts*(1000*dt)^2
thermostatchain = [Thermostat(Ts, Qs, 0.0, 0.0) for i in 1:20]
for i in 2:length(thermostatchain)
thermostatchain[i].Q=Qs/natom/3
end

barostat=Barostat(Ps,Ws,cell.Volume,0.0)

I::BigFloat= 1.0  
    
E2 = I / 6.0
E4 = E2 / 20.0
E6 = E4 / 42.0
E8 = E6 / 72.0

for step in 1:1000000

    natom=length(cell.atoms)
    
    W=barostat.W
    
    Nhcpisoint!(cell,interaction,thermostatchain,barostat,dt)
    xlogv=1/3*log(barostat.V)
    vlogv=barostat.Pv
    initial_volume=cell.Volume
    # println(xlogv)
    for i in 1:natom
    fi=cell_forcei(cell,interaction,i)
    mi=cell.atoms[i].mass
    cell.atoms[i].momentum.+=dt2*fi
    end
    AA=exp(dt2*vlogv)
    AA2=AA*AA
    # println(vlogv)
    arg2=(vlogv*dt2)^2
    poly=(((E8*arg2+E6)*arg2+E4)*arg2+E2)+I
    BB=AA*poly*dt
    invltv=inv(cell.lattice_vectors)
    for i in 1:natom
        # println("atom:$i,pos:$(cell.atoms[i].position),$AA2,$((BB)*(invltv*cell.atoms[i].momentum/cell.atoms[i].mass))")
        cell.atoms[i].position.=cell.atoms[i].position*AA2+(BB)*(invltv*cell.atoms[i].momentum/cell.atoms[i].mass)
        # println("atom:$i,pos:$(cell.atoms[i].position),$AA2,$((BB)*(invltv*cell.atoms[i].momentum/cell.atoms[i].mass))")
    end
    
    xlogv=xlogv+vlogv*dt
    cell.Volume=exp(3*xlogv)
    cell.lattice_vectors=cell.lattice_vectors*(cell.Volume/initial_volume)^(1/3)
    barostat.V=cell.Volume
    barostat.Pv=vlogv
    for i in 1:natom
    fi=cell_forcei(cell,interaction,i)
    mi=cell.atoms[i].mass
    cell.atoms[i].momentum.+=dt2*fi
    end
    
    # println("step: ",step," Temp: ",cell_temp(cell)," Pressure: ",pressure_int(cell,interaction)," Volume: ",cell.Volume)
    
    Nhcpisoint!(cell,interaction,thermostatchain,barostat,dt)
    
    cp=cell.copy
    for i in 1:natom 
        for j in 1:3
            cell.atoms[i].position[j]=mod(cell.atoms[i].position[j],cp[j])
        end
    end
    
    




pint=pressure_int(cell,interaction)
T=cell_temp(cell)
    if mod(step,100)==0
    println("step: $step, Temp: $T Pressure: $pint,Volume: $(cell.Volume),Pv:$(barostat.Pv)")
    writedlm(io, [step, T, pint]')
    write(iojl, "cell_$step", cell)
    end

end
end
end

