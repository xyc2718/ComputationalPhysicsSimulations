"""
Water Model of TIP3P
https://docs.lammps.org/Howto_tip3p.html

As Coulomb Potential is Long Range, we need to use Ewald Summation OR PPPM Grid to calculate it in Periodic Boundary Condition.
The Ewald Simulation is not implemented in this code yet.
So the Energy of NVE ensemble is not conserved with PPP if the cutoff is longer than half of the box.
You need to selcet a good cutoff of Coulomb Potential when simmulate in NVT/NVE ensemble.The Defalut cutoff for Coulomb is 4.0A.
"""
module WaterModel
using ..Model
using StaticArrays
using LinearAlgebra

export TIP3P,getparatip3p,EOH,FOH,EHOH,FHOH,LJE,LJF,CoulombE,CoulombF,getparaqSPC

function getparatip3p(;maxcutoff=10.0)
    kk=0.0433634
    para=Dict(
        "kOH" => kk*450,    # eV/A
        "kHOH" => 55.0*kk,   # [m]/amu
        "rOH" => 0.9572,       # amu
        "theta0" => 104.52*pi/180,      # GPa/[p]
        "h" => 6.582119281e-4 ,     # eV*ps
        "eOO"=>0.1521*kk,
        "sigmaOO"=>3.1507, #A
        "eOH"=>0.0836*kk,
        "sigmaOH"=>1.7753,
        "eHH"=>0.0460*kk,
        "sigmaHH"=>0.4,
        "eO"=>-0.834,
        "eH"=>0.417,
        "ctLJOO"=>3.1507*3.0,
        "ctLJHH"=>3.0*0.4,
        "ctLJOH"=>1.78*3.0,
        "ctCoulomb"=>6.0, #A
        "ct"=>6.0
    )
    if para["ctLJOO"]>maxcutoff
        para["ctLJOO"]=maxcutoff
    end
    if para["ctLJHH"]>maxcutoff
        para["ctLJHH"]=maxcutoff
    end
    if para["ctLJOH"]>maxcutoff
        para["ctLJOH"]=maxcutoff
    end
    if para["ctCoulomb"]>maxcutoff
        para["ctCoulomb"]=maxcutoff
    end
    if para["ct"]>maxcutoff
        para["ct"]=maxcutoff
    end
    return para

    
end

"""
FIXME:该势能参数存疑
Ref:Paesani, F., Zhang, W., Case, D. A., Cheatham, T. E., & Voth, G. A. (2006). An accurate and simple quantum model for liquid water. The Journal of Chemical Physics, 125(18), 184507. https://doi.org/10.1063/1.2386157
它的水分子theta0是112度，似乎是要考虑内部的相互作用的(?)，不能直接把参数扔到tip3p里面
然后文中的参数表没有oH,HH的LJ，但是公式里面又有 ϵ_ij 所以OH是否存在LJ存疑
"""
function getparaqSPC()
    kk=0.0433634
    return Dict(
        "kOH" => kk*1059.162,    # eV/A
        "kHOH" => 75.90*kk,   # [m]/amu
        "rOH" => 1.000,       # amu
        "theta0" => 112.0*pi/180,      # GPa/[p]
        "h" => 6.582119281e-4 ,     # eV*ps
        "eOO"=>0.1554252*kk,
        "sigmaOO"=>3.165492, #A
        "eOH"=>0.0*kk,
        "sigmaOH"=>1.7753,
        "eHH"=>0.0*kk,
        "sigmaHH"=>0.4,
        "eO"=>-0.84,
        "eH"=>0.42,
        "ctLJOO"=>2.0*3.15,
        "ctLJHH"=>3.0*0.4,
        "ctLJOH"=>1.78*2.0,
        "ctCoulomb"=>3.0, #A
        "ct"=>4.0
    )
    
end



function EOH(r::SVector{3,Float64})
    pare=getparatip3p()
    k=pare["kOH"]
    r0=pare["rOH"]
    return k*(norm(r)-r0)^2
end
function FOH(r::SVector{3,Float64})
    nr=norm(r)
    pare=getparatip3p()
    k=pare["kOH"]
    r0=pare["rOH"]
    return (2*k*(nr-r0)*(r/nr),-2*k*(nr-r0)*(r/nr))
end



function EHOH(r1::SVector{3,Float64},r2::SVector{3,Float64})
    
    pare=getparatip3p()
    k=pare["kHOH"]
    theta0=pare["theta0"]
    nr1=norm(r1)
    nr2=norm(r2)
    cs=LinearAlgebra.dot(r1,r2)/nr1/nr2
    if cs>1.0
        cs=1.0
    end
    if cs<-1.0
        cs=-1.0
    end
    theta=acos(cs)
    return k*(theta-theta0)^2
end
function FHOH(r1::SVector{3,Float64},r2::SVector{3,Float64})
    pare=getparatip3p()
    k=pare["kHOH"]
    theta0=pare["theta0"]
    nr1=norm(r1)
    nr2=norm(r2)
    cs=LinearAlgebra.dot(r1,r2)/nr1/nr2
    n=LinearAlgebra.cross(r1,r2)
    t1=LinearAlgebra.cross(r1,n)
    t2=LinearAlgebra.cross(n,r2)
    t1=t1/norm(t1)
    t2=t2/norm(t2)
    if cs>1.0
        cs=1.0
    end
    if cs<-1.0
        cs=-1.0
    end
    theta=acos(cs)
    return (-2*k*(theta-theta0)*t1/nr1,-2*k*(theta-theta0)*t2/nr2)
end



function LJE(r::Float64,eps::Float64,sig::Float64)
    nr=norm(r)
        return 4*eps*(sig^12/nr^12-sig^6/nr^6)
end

function LJF(r::SVector{3,Float64},eps::Float64,sig::Float64)::SVector{3,Float64}
    nr=norm(r)
    return 24*eps*(2*sig^12/nr^14-sig^6/nr^8)*r
end

function CoulombE(r::Float64,e1::Float64,e2::Float64)
    para=getpara()
    K=para["K"]
    return K*e1*e2/r
end


function CoulombF(r::SVector{3,Float64},e1::Float64,e2::Float64)::SVector{3,Float64}
    nr=norm(r)
    para=getpara()
    K=para["K"]
    return e1*e2/nr^3*r*K
end

"""
return the TIP3P Interactions
you need to map molecule to a cell and get water(typeof Molecule) from it
cutCoulomb is the cutoff of Coulomb Potential, if it is less than 0, the default value will be used.

Also can use other paraments by change keyword para
"""
function TIP3P(water::Molecule;cutCoulomb::Float64=-1.0,para=nothing,maxcutoff=10.0)
    if para===nothing
        paratip3p=getparatip3p(maxcutoff=maxcutoff)
    else
        paratip3p=para
    end
    
    conOH=Vector{Vector{Int}}([])
    for cn in water.connection
        for i in 2:length(cn)
        push!(conOH,[cn[1],cn[i]])
        end
    end
    bondOH=Bond(conOH,EOH,FOH)
    conHOH=water.connection
    natom= maximum(vcat(water.connection...))
    Oid=[i for i in 1:natom if mod(i,3)==1]
    Hid=[i for i in 1:natom  if mod(i,3)!=1 ]

    LJOO=Vector{Vector{Int}}([])
    LJOH=Vector{Vector{Int}}([])
    LJHH=Vector{Vector{Int}}([]) 
    """
    !!!注意:
    静电力也不计入分子内的相互作用，故对tip3p来说其邻居列表应与LJ相同
    """
    # CoulombOO=Vector{Vector{Int}}([])
    # CoulombOH=Vector{Vector{Int}}([])
    # CoulombHH=Vector{Vector{Int}}([])
    for i in 1:natom
        if i in Oid
            push!(LJOO,filter(x->x!=i,Oid))
            push!(LJOH,filter(x->(x!=i+1)&&(x!=i+2),Hid))
            # push!(CoulombOH,filter(x->x!=i,Hid))
            # push!(CoulombOO,filter(x->x!=i,Oid))
        else
            push!(LJOO,[])
            if mod(i,3)==2
                push!(LJOH,filter(x->x!=i-1,Oid))
            end
            if mod(i,3)==0
                push!(LJOH,filter(x->x!=i-2,Oid))
            end

            # push!(CoulombOH,Oid)
            # push!(CoulombOO,[])
        end
    end


    for i in 1:natom
        if i in Hid
            if mod(i,3)==2
                push!(LJHH,filter(x->(x!=i)&&(x!=i+1),Hid))
            end
            if mod(i,3)==0
                push!(LJHH,filter(x->(x!=i)&&(x!=i-1),Hid))
            end
            # push!(CoulombHH,filter(x->x!=i,Hid))
        else
            push!(LJHH,[])
            # push!(CoulombHH,[])
        end

    end
    CoulombHH=deepcopy(LJHH)
    CoulombOH=deepcopy(LJOH)
    CoulombOO=deepcopy(LJOO)
    # println(conOH)
    # println(conHOH)
    # println(Oid)
    # println(natom)
    # println(LJOO)
    # println(LJOH)
    # println(LJHH)
    # println(CoulombOO)
    # println(CoulombOH)
    # println(CoulombHH)
    

    LJOOE(r::Float64)=LJE(r,paratip3p["eOO"],paratip3p["sigmaOO"])
    LJOHE(r::Float64)=LJE(r,paratip3p["eOH"],paratip3p["sigmaOH"])
    LJHHE(r::Float64)=LJE(r,paratip3p["eHH"],paratip3p["sigmaHH"])
    LJOOF(r::SVector{3,Float64})=LJF(r,paratip3p["eOO"],paratip3p["sigmaOO"])
    LJOHF(r::SVector{3,Float64})=LJF(r,paratip3p["eOH"],paratip3p["sigmaOH"])
    LJHHF(r::SVector{3,Float64})=LJF(r,paratip3p["eHH"],paratip3p["sigmaHH"])
    CoulombOOE(r::Float64)=CoulombE(r,paratip3p["eO"],paratip3p["eO"])
    CoulombOHE(r::Float64)=CoulombE(r,paratip3p["eO"],paratip3p["eH"])
    CoulombHHE(r::Float64)=CoulombE(r,paratip3p["eH"],paratip3p["eH"])
    CoulombOOF(r::SVector{3,Float64})=CoulombF(r,paratip3p["eO"],paratip3p["eO"])
    CoulombOHF(r::SVector{3,Float64})=CoulombF(r,paratip3p["eO"],paratip3p["eH"])
    CoulombHHF(r::SVector{3,Float64})=CoulombF(r,paratip3p["eH"],paratip3p["eH"])
    ct=paratip3p["ctLJOO"]
    interLJOO=Interaction(LJOOE,LJOOF,ct,0.1*ct)
    NeighborLJOO=Neighbor(LJOO)
    ct=paratip3p["ctLJOH"]
    interLJOH=Interaction(LJOHE,LJOHF,ct,0.1*ct)
    NeighborLJOH=Neighbor(LJOH)
    ct=paratip3p["ctLJHH"]
    interLJHH=Interaction(LJHHE,LJHHF,ct,0.1*ct)
    NeighborLJHH=Neighbor(LJHH)
    if cutCoulomb<0
        ct=paratip3p["ctCoulomb"]
    else
        ct=cutCoulomb
    end
    interCoulombOO=Interaction(CoulombOOE,CoulombOOF,ct,0.1*ct)
    NeighborCoulombOO=Neighbor(CoulombOO)
    interCoulombOH=Interaction(CoulombOHE,CoulombOHF,ct,0.1*ct)
    NeighborCoulombOH=Neighbor(CoulombOH)
    interCoulombHH=Interaction(CoulombHHE,CoulombHHF,ct,0.1*ct)
    NeighborCoulombHH=Neighbor(CoulombHH)


    angleHOH=Angle(conHOH,EHOH,FHOH)

    nb=Vector{Neighbor}([Neighbor(),Neighbor(),NeighborLJOO,NeighborLJOH,NeighborLJHH,NeighborCoulombOO,NeighborCoulombOH,NeighborCoulombHH])
    interactionlist=Vector{AbstractInteraction}([bondOH,angleHOH,interLJOO,interLJOH,interLJHH,interCoulombOO,interCoulombOH,interCoulombHH])
    interactions=Interactions(interactionlist,nb)


    # nb=Vector{Neighbor}([NeighborLJOO,NeighborLJOH,NeighborLJHH])
    # interactionlist=Vector{AbstractInteraction}([interLJOO,interLJOH,interLJHH])
    # nb=Vector{Neighbor}([NeighborLJOO])
    # interactionlist=Vector{AbstractInteraction}([interLJOO])
    #   nb=Vector{Neighbor}([NeighborLJHH])
    # interactionlist=Vector{AbstractInteraction}([interLJHH])
    # interactions=Interactions(interactionlist,nb)
    # println([NeighborLJOO,NeighborLJHH])

    # nb=Vector{Neighbor}([NeighborCoulombOO,NeighborCoulombOH,NeighborCoulombHH])
    # interactionlist=Vector{AbstractInteraction}([interCoulombOO,interCoulombOH,interCoulombHH])

    # interactions=Interactions(interactionlist,nb)
    # nb=Vector{Neighbor}([Neighbor(),Neighbor()])
    # interactionlist=Vector{AbstractInteraction}([bondOH,angleHOH])
    # interactions=Interactions(interactionlist,nb)

    # nb=Vector{Neighbor}([NeighborLJHH])
    # interactionlist=Vector{AbstractInteraction}([interLJHH])
    # interactions=Interactions(interactionlist,nb)
    # println(interCoulombOO.cutoff)
    # println(NeighborCoulombOO)
    # println(NeighborLJHH)
    @info "Initialized with TIP3P Interactions with Coulomb cutoff $cutCoulomb,maxcutoff $maxcutoff,\n parameters:$paratip3p"
    return interactions
end



end