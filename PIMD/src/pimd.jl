"""
PIMD and TRPIMD simulation
pimdStep! for PIMD simulation
pimdLangevinStep! for TRPIMD simulation with Langevin thermostat
you need to map cell to BeadCell first with map2bead

reference:
Ceriotti, M., Parrinello, M., Markland, T. E., & Manolopoulos, D. E. (2010). Efficient stochastic thermostatting of path integral molecular dynamics. The Journal of Chemical Physics, 133(12), 124104. https://doi.org/10.1063/1.3489925
"""
module PIMD
using ..Model
using ..MD
using Base.Threads
using Statistics
using  LinearAlgebra
export initBeadCell!,map2bead,pimdL0Step!,pimdLgammaStep!,pimdLvStep!,pimdStep!,pimdLangevinStep!,updateBeadCell!,cell_Ek,get_bead_z,get_bead_z0,apply_PBC_BDC!,cell_Ek1,cell_Ek0


"""
return pl,ql for bead cell p[1,:]==p[end,:],q[1,:]==q[end,:]->cells[end]    p[2,:]->cells[1],ql is real position
"""
function get_bead_z(bdc::BeadCell)
    nbeads=bdc.nbeads
    natom=length(bdc.cells[1].atoms)
    ql=Matrix{Float64}(undef,nbeads+1,natom*3)
    pl=Matrix{Float64}(undef,nbeads+1,natom*3)
    ltv=bdc.cells[1].lattice_vectors
    for i in 2:nbeads+1
        celli=bdc.cells[i-1]
        for k in eachindex(celli.atoms)
            ql[i,3k-2:3k]=ltv*celli.atoms[k].position
            pl[i,3k-2:3k]=celli.atoms[k].momentum
        end
    end
    celli=bdc.cells[end]
    for k in eachindex(celli.atoms)
        ql[1,3k-2:3k]=ltv*celli.atoms[k].position
        pl[1,3k-2:3k]=celli.atoms[k].momentum
    end
    return pl,ql
end

"""
return pl,ql for bead cell p[1,:]==p[end,:],q[1,:]==q[end,:]->cells[end]    p[2,:]->cells[1],ql is relative position
"""
function get_bead_z0(bdc::BeadCell)
    nbeads=bdc.nbeads
    natom=length(bdc.cells[1].atoms)
    ql=Matrix{Float64}(undef,nbeads+1,natom*3)
    pl=Matrix{Float64}(undef,nbeads+1,natom*3)
    for i in 2:nbeads+1
        celli=bdc.cells[i-1]
        for k in eachindex(celli.atoms)
            ql[i,3k-2:3k]=celli.atoms[k].position
            pl[i,3k-2:3k]=celli.atoms[k].momentum
        end
    end
    celli=bdc.cells[end]
    for k in eachindex(celli.atoms)
        ql[1,3k-2:3k]=celli.atoms[k].position
        pl[1,3k-2:3k]=celli.atoms[k].momentum
    end
    return pl,ql
end

"""
Transform Matrix to Norm position
"""
function Cmatk(bdc::BeadCell,T)
    N=bdc.nbeads
    cmat=zeros(Float64,N,N)
    for j in 1:N
        for k in 0:N-1
            if k==0
                cjk=sqrt(1/N)
            elseif 1<=k<=N/2-1
                cjk=sqrt(2/N)*cos(2*pi*k*j/N)
            elseif k==N/2
                cjk=sqrt(1/N)*(-1)^(j)
            elseif N/2+1<=k<=N-1
                cjk=sqrt(2/N)*sin(2*pi*k*j/N)
                # println("j=$j k=$k cjk=$cjk")
            else
                throw(ArgumentError("k out of range"))
            end
            cmat[j,k+1]=cjk
        end
    end
    return cmat
end

function initBeadCell!(bdc::BeadCell,T::Float64=1.0)
    cm=Cmatk(bdc,T)
    bdc.cmat=cm

end

function initBeadCell!(bdc::BeadCell,interaction::AbstractInteraction,T::Float64=1.0)
    cm=Cmatk(bdc,T)
    bdc.cmat=cm
    for cell in bdc.cells
        update_rmat!(cell)
        update_fmat!(cell,interaction)
    end

end
"""
Map a UnitCell to BeadCell with copy to nbeads,r will give it a circle displacement
"""
function map2bead(cell::UnitCell,interaction::AbstractInteraction,nbeads::Int,T=1.0;r=0.00)
    cells=Vector{UnitCell}(undef,nbeads)
    for i in 1:nbeads
        celli=deepcopy(cell)
        for k in eachindex(cell.atoms)
            celli.atoms[k].position+=r*[sin(2*pi*i/nbeads),cos(2*pi*i/nbeads),0.0]
        end
        cells[i]=celli
    end
    bdc=BeadCell(cells)
    initBeadCell!(bdc,interaction,T)
    return bdc
end

function map2bead(cell::UnitCell,nbeads::Int,T=1.0;r=0.00)
    cells=Vector{UnitCell}(undef,nbeads)
    for i in 1:nbeads
        celli=deepcopy(cell)
        for k in eachindex(cell.atoms)
            celli.atoms[k].position+=r*[sin(2*pi*i/nbeads),cos(2*pi*i/nbeads),0.0]
        end
        cells[i]=celli
    end
    bdc=BeadCell(cells)
    initBeadCell!(bdc,T)
    return bdc
end



function updateBeadCell!(bdc::BeadCell,pl::Matrix{Float64},ql::Matrix{Float64})
    nbeads=bdc.nbeads
    invlt=inv(bdc.cells[1].lattice_vectors)
    # @threads 
for i in 2:nbeads+1
        celli=bdc.cells[i-1]
        for k in eachindex(celli.atoms)
            celli.atoms[k].momentum=pl[i,3k-2:3k]
            celli.atoms[k].position=invlt*ql[i,3k-2:3k]
        end
    end

end

function updateBeadCell0!(bdc::BeadCell,pl::Matrix{Float64},ql::Matrix{Float64})
    nbeads=bdc.nbeads
    # @threads 
for i in 2:nbeads+1
        celli=bdc.cells[i-1]
        for k in eachindex(celli.atoms)
            celli.atoms[k].momentum=pl[i,3k-2:3k]
            celli.atoms[k].position=ql[i,3k-2:3k]
        end
    end

end

"""
L for norm mode evolution
"""
function pimdL0Step!(bdc::BeadCell,dt::Float64,T::Float64=1.0)
    para=getpara()
    kb=para["kb"]
    h=para["h"]
    
    N=bdc.nbeads
    m = [atom.mass for atom in bdc.cells[1].atoms for _ in 1:3]
    natom=length(bdc.cells[1].atoms)
    betan=1/N/kb/T
    wn=1.0/h/betan
    pl,ql=get_bead_z(bdc)
    ql0=deepcopy(ql)
    pl0=deepcopy(pl)
    ql0[1:end-1,:]=transpose(transpose(ql[1:end-1,:])*bdc.cmat)
    pl0[1:end-1,:]=transpose(transpose(pl[1:end-1,:])*bdc.cmat)
    ql0[end,:]=ql0[1,:]
    pl0[end,:]=pl0[1,:]
    I=BigFloat(1.0)
    E1=-I/6
    E2=I/120
    @threads for k in 1:N
        wk=2*wn*sin((k-1)*pi/N)
        pl0[k,:]=pl0[k,:].*cos(wk*dt)+m.*ql0[k,:]*(-wk).*sin(wk*dt)
        # println("$(ql0)")
        if wk<0.01
        ql0[k,:]=pl0[k,:].*dt*(I+wk^2*dt^2*(E1+E2*wk^2*dt*2))./m+ql0[k,:].*cos(wk*dt)
        else
            ql0[k,:]=pl0[k,:].*sin(wk*dt)./m/wk+ql0[k,:].*cos(wk*dt)
        end
        # println("$(ql0)")
    end
    pl0[end,:]=pl0[1,:]
    ql0[end,:]=ql0[1,:]
    pl[2:end,:]=bdc.cmat*pl0[1:end-1,:]
    ql[2:end,:]=bdc.cmat*ql0[1:end-1,:]
    pl[1,:]=pl[end,:]
    ql[1,:]=ql[end,:]
    updateBeadCell!(bdc,pl,ql)
    # return pl,ql
end

"""
Langevin thermostat apply on norm mode evolution
"""
function pimdLgammaStep!(bdc::BeadCell,dt::Float64,T::Float64=1.0;t0::Float64=0.1)
    para=getpara()
    kb=para["kb"]
    h=para["h"]
    N=bdc.nbeads
    m = [atom.mass for atom in bdc.cells[1].atoms for _ in 1:3]
   
    natom=length(bdc.cells[1].atoms)
    betan=1/N/kb/T
    wn=1.0/h/betan
    pl,ql=get_bead_z(bdc)
    pl0=deepcopy(pl)
    pl0[1:end-1,:]=transpose(transpose(pl[1:end-1,:])*bdc.cmat)
    pl0[end,:]=pl0[1,:]
    @threads for k in 1:N
        wk=2*wn*sin((k-1)*pi/N)
        if k==1
            gammak=1/t0
        else
            gammak=2*wk
        end
        c1k=exp(-gammak*dt*0.5)
        c2k=sqrt(1-c1k^2)
        # println("1",size(m))
        # println(m)
        # println("2",size(pl0[k,:]))
        pl0[k,:]=pl0[k,:].*c1k.+sqrt.(m./betan).*c2k.*randn(3*natom)
    end
    pl0[end,:]=pl0[1,:]
    pl[2:end,:]=bdc.cmat*pl0[1:end-1,:]
    pl[1,:]=pl[end,:]
    updateBeadCell!(bdc,pl,ql)
end

"""
L for V(q) evolution
"""
function pimdLvStep!(bdc::BeadCell,dt::Float64,interactions::AbstractInteraction)
    nbeads=bdc.nbeads

    apply_PBC_BDC!(bdc,interactions)
    # @threads 这里使用@threads 大概能提速一倍左右，但是对bead数较多时容易导致内存溢出
   for nb in 1:nbeads
        celli=bdc.cells[nb]
        # apply_PBC!(celli,interactions)
        for i in eachindex(celli.atoms)
            atomi=celli.atoms[i]
            # println(cell_forcei(celli,interactions,i))
            atomi.momentum+=0.5*dt*cell_forcei(celli,interactions,i)
        end
    end
end

"""
PIMD step with no thermostat
"""
function pimdStep!(bdc::BeadCell,dt::Float64,T::Float64,interactions::AbstractInteraction)
    pimdLvStep!(bdc,dt,interactions)
    pimdL0Step!(bdc,dt,T)
    pimdLvStep!(bdc,dt,interactions)
end

"""
PIMD step with Langevin thermostat
"""
function pimdLangevinStep!(bdc::BeadCell,dt::Float64,T::Float64,interactions::AbstractInteraction;t0::Float64=0.1)
    pimdLgammaStep!(bdc,dt,T,t0=t0)
    pimdLvStep!(bdc,dt,interactions)
    pimdL0Step!(bdc,dt,T)
    pimdLvStep!(bdc,dt,interactions)
    pimdLgammaStep!(bdc,dt,T,t0=t0)
end

"""
return the kinetic energy of bead cell by calculate veri term
"""
function cell_Ek(bdc::BeadCell,interactions::AbstractInteraction,args...)
    para=getpara()
    kb=para["kb"]
    Ek0=0.0
    N=bdc.nbeads
    mc = [atom.mass for atom in bdc.cells[1].atoms for _ in 1:3]
    natom=length(bdc.cells[1].atoms)
    pl,ql=get_bead_z(bdc)
    qlm=Statistics.mean(ql[2:end,:],dims=1)
    # println("qlm=$(qlm[1:10])")
    # println("ql=$(ql[:,1:10])")

    for i in 1:natom
        for j in 1:N
            fi=cell_forcei(bdc.cells[j],interactions,i)
            Ek0-= LinearAlgebra.dot((ql[j+1,3*i-2:3*i]-qlm[3*i-2:3*i]),fi)
            # println(LinearAlgebra.dot((ql[j+1,3*i-2:3*i]-qlm[3*i-2:3*i]),fi))
        end        
    end
    m=repeat(mc',N,1)
    Ek1=0.5*Statistics.mean((pl[2:end,:].^2)./m)/N
    # Ek1=1.5*natom*kb*Ts
    # return 3*natom*kb*Ts/2+Ek0/2/N
    return Ek1+Ek0/2/N
    
end

function cell_Ek(cell::UnitCell,args...)
    Ek=0.0
    for atom in cell.atoms
        Ek+=0.5*norm(atom.momentum)^2/atom.mass
    end
    return Ek
    
end

"""
veri term
"""
function cell_veri(bdc::BeadCell,interactions::AbstractInteraction)
    para=getpara()
    kb=para["kb"]
    Ek0=0.0
    N=bdc.nbeads
    natom=length(bdc.cells[1].atoms)
    pl,ql=get_bead_z(bdc)
    qlm=Statistics.mean(ql[2:end,:],dims=1)
    for i in 1:natom
        for j in 1:N
            fi=cell_forcei(bdc.cells[j],interactions,i)
            Ek0-= LinearAlgebra.dot((ql[j+1,3*i-2:3*i]-qlm[3*i-2:3*i]),fi)
        end        
    end
    return Ek0/2/N
end



"""
return the kinetic energy of bead cell by calculate the harmonic oscillator term，this will lead to larger fluctuate
"""
function cell_Ek0(bdc::BeadCell,interactions::AbstractInteraction,Ts::Float64)
    para=getpara()
    kb=para["kb"]
    h=para["h"]
    Ek0=0.0
   
    N=bdc.nbeads
    betan=1/N/kb/Ts
    wn=1.0/h/betan
    N=bdc.nbeads
    m = [atom.mass for atom in bdc.cells[1].atoms for _ in 1:3]
    natom=length(bdc.cells[1].atoms)
    pl,ql=get_bead_z(bdc)
        for j in 1:N-1
        
            Ek0-=sum((((ql[j+2,:]-ql[j+1,:]).^2).*m).*(wn^2/2 ))
        end        
    return 3*natom*N*kb*Ts/2+Ek0/2
end

"""
apply PBC on bead cell,this will depend on the center of the Ring
"""
function apply_PBC_BDC!(bdc::BeadCell,interactions::AbstractInteraction)
    if  length(bdc.cells[1].molecule.atoms)==0
        cpc=bdc.cells[1].copy
        pl,ql=get_bead_z0(bdc)
        qlm=Statistics.mean(ql[2:end,:],dims=1)
        for i in 1:length(bdc.cells[1].atoms)
                for k in 1:3
                    if qlm[3i-3+k]>cpc[k]
                        ql[:,3i-3+k].-=2*cpc[k]
                    elseif qlm[3i-3+k]<-cpc[k]
                        ql[:,3i-3+k].+=2*cpc[k]
                    end
            end
        end
        updateBeadCell0!(bdc,pl,ql)
    


        else
            pl,ql=get_bead_z0(bdc)
            qlm=Statistics.mean(ql[2:end,:],dims=1)
            a,b,c=bdc.cells[1].copy
            for cn in bdc.cells[1].molecule.connection
                ct=cn[1]
                rct=qlm[ct*3-2:ct*3]
                ix,iy,iz=0,0,0
                if rct[1]>a
                    ix=1
                end
                if rct[1]<-a
                    ix=-1
                end
                if rct[2]>b
                    iy=1
                end
                if rct[2]<-b
                    iy=-1
                end
                if rct[3]>c
                    iz=1
                end
                if rct[3]<-c
                    iz=-1
                end
                for pb in 1:length(cn)
                    for cell in bdc.cells
                        r=cell.atoms[pb].position
                        r[1]=r[1]-ix*2*a
                        r[2]=r[2]-iy*2*b
                        r[3]=r[3]-iz*2*c
                        cell.atoms[pb].position=r
                    end
                end
            end
        end
        for cell in bdc.cells
            update_rmat!(cell)
            update_fmat!(cell,interactions)
        end
    
end






end