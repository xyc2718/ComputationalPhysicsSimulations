module WaterModel
using ..Model
using StaticArrays
using LinearAlgebra

export TIP3P,getparatip3p

function getparatip3p()
    kk=0.0433634
    return Dict(
        "kOH" => kk*450,    # eV/A
        "kHOH" => 55.0*kk,   # [m]/amu
        "rOH" => 0.9572,       # amu
        "theta0" => 104.52*pi/180,      # GPa/[p]
        "h" => 6.582119281e-4 ,     # eV*ps
        "eOO"=>0.1521*kk,
        "sigmaOO"=>3.1507,
        "eOH"=>0.0836*kk,
        "sigmaOH"=>1.7753,
        "eHH"=>0.0460*kk,
        "sigmaHH"=>0.4,
        "eO"=>-0.834,
        "eH"=>0.417,
        "ct"=>3.0,
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
    return -e1*e2/nr^3*r
end

function TIP3P(water::Molecule)
    
    conOH=Vector{Vector{Int}}([])
    for cn in water.connection
        for i in 2:length(cn)
        push!(conOH,[cn[1],cn[i]])
        end
    end
    paratip3p=getparatip3p()
    bondOH=Bond(conOH,EOH,FOH)
    conHOH=water.connection
    Oid=[i for i in 1:length(water.atoms) if mod(i,3)==0]
    Hid=[i for i in 1:length(water.atoms) if mod(i,3)!=0 ]

    LJOO=Vector{Vector{Int}}([])
    LJOH=Vector{Vector{Int}}([])
    LJHH=Vector{Vector{Int}}([])
    CoulombOO=Vector{Vector{Int}}([])
    CoulombOH=Vector{Vector{Int}}([])
    CoulombHH=Vector{Vector{Int}}([])
    for i in Oid
        push!(LJOO,filter(x->x!=i,Oid))
        push!(LJOH,filter(x->x!=i,Hid))
        push!(CoulombOH,filter(x->x!=i,Hid))
        push!(CoulombOO,filter(x->x!=i,Oid))
    end

    for i in Hid
        if mod(i,3)==1
            push!(LJHH,filter(x->(x!=i)&&(x!=i+1),Hid))
        end
        if mod(i,3)==2
            push!(LJHH,filter(x->(x!=i)&&(x!=i-1),Hid))
        end
        push!(CoulombHH,filter(x->x!=i,Hid))

    end
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
    ct=paratip3p["ct"]
    interLJOO=Interaction(LJOOE,LJOOF,ct,0.1)
    NeighborLJOO=Neighbor(LJOO)
    interLJOH=Interaction(LJOHE,LJOHF,ct,0.1)
    NeighborLJOH=Neighbor(LJOH)
    interLJHH=Interaction(LJHHE,LJHHF,ct,0.1)
    NeighborLJHH=Neighbor(LJHH)
    interCoulombOO=Interaction(CoulombOOE,CoulombOOF,ct,0.1)
    NeighborCoulombOO=Neighbor(CoulombOO)
    interCoulombOH=Interaction(CoulombOHE,CoulombOHF,ct,0.1)
    NeighborCoulombOH=Neighbor(CoulombOH)
    interCoulombHH=Interaction(CoulombHHE,CoulombHHF,ct,0.1)
    NeighborCoulombHH=Neighbor(CoulombHH)


    angleHOH=Angle(conHOH,EHOH,FHOH)

    nb=Vector{Neighbor}([Neighbor(),Neighbor(),NeighborLJOO,NeighborLJOH,NeighborLJHH,NeighborCoulombOO,NeighborCoulombOH,NeighborCoulombHH])
    interactionlist=Vector{AbstractInteraction}([bondOH,angleHOH,interLJOO,interLJOH,interLJHH,interCoulombOO,interCoulombOH,interCoulombHH])
    interactions=Interactions(interactionlist,nb)
    return interactions
end

end