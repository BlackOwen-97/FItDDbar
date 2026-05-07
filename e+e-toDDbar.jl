using Base.Threads
using LinearAlgebra
using Printf, Plots
using QuadGK
include("General.jl")
include("NumIntegrate.jl")
const mD  = 1864.84   # D 介子质量
const mDs = 1968.34   # Ds 介子质量
const mee = 0.511


const Alpha = 1.0/137.036   # 精细结构常数
const hc    = 197.3269804   # MeV·fm
const cs    = 3.893793656e5 


# LSx  : 积分节点
# LSw  : 积分权重
# Nnode: 节点数（25）
# J    : 总角动量（对于 1P1，J=1）
# Iso  : 同位旋（0 或 1）
# f0   : 介子衰变常数
# gD   : D*Dπ 耦合常数
# alp  : 截断参数


function Cutf(p, p1, lam)
    return exp(-(p^6 + p1^6) / lam^6)
end

function Cutborn(k, lam_born=900.0)
    return exp(-(k^6) / lam_born^6)
end


function qsq(p, p1, x)
    return p^2 + p1^2 - 2*p*p1*x
end



function OTBEV_DD(lam)
    @inline is_pcm(i) = (i == 25 || i == 50)

    LSx,LSw=LSpoints(Nnode,700)

    LSx1 = [LSx; 0.0]; LSx1 = vcat(LSx1, LSx1, LSx1, LSx1)  
    LSx2 = [LSx; 0.0]; LSx2 = vcat(LSx2, LSx2, LSx2, LSx2)


    V_OBE_DD   = (p, p1, x) -> DDbarobe(p, p1, x, Iso, gD, alp, f0, mD)
    V_TBE_DD   = (p, p1, x) -> DDbartbe(sqrt(qsq(p, p1, x)), Iso, gD, alp, f0)

    V_OBE_NL   = (p, p1, x) -> DDbarDsDsobe(p, p1, x, Iso, gD, alp, f0, mD, mDs)
    V_TBE_NL   = (p, p1, x) -> DDbarDsDstbe(sqrt(qsq(p, p1, x)), Iso, gD, alp, f0)

    V_OBE_Ds   = (p, p1, x) -> DsDsbarobeDs(p, p1, x, Iso, gD, alp, f0, mDs)
    V_TBE_Ds   = (p, p1, x) -> DsDsbartbeDs(sqrt(qsq(p, p1, x)), Iso, gD, alp, f0)

    M11 = zeros(ComplexF64, 50, 50)   # DDbar → DDbar
    M12 = zeros(ComplexF64, 50, 50)   # DDbar → DsDsbar
    M22 = zeros(ComplexF64, 50, 50)   # DsDsbar → DsDsbar

    @inline function Lcode_from_ij(i, j)
        a = (i-1) ÷ 25
        b = (j-1) ÷ 25
        return (a==0 && b==0) ? 4 :
               (a==0 && b==1) ? 6 :
               (a==1 && b==0) ? 5 : 3
    end

    # M11: DDbar → DDbar
    @threads for i in 1:50
        @inbounds for j in 1:50
            if is_pcm(i) || is_pcm(j); continue; end
            Lc = Lcode_from_ij(i, j)
            p  = LSx1[i]
            p1 = LSx1[j]
            M11[i,j] = PartialWave(V_OBE_DD, p, p1, J, Lc, Nnode) * Cutf(p, p1, lam)
        end
    end

    # M12: DDbar → DsDsbar
    @threads for i in 1:50
        @inbounds for j in 1:50
            if is_pcm(i) || is_pcm(j); continue; end
            Lc = Lcode_from_ij(i, j)
            p  = LSx1[i]
            p1 = LSx2[j]
            M12[i,j] = PartialWave(V_OBE_NL, p, p1, J, Lc, Nnode) * Cutf(p, p1, lam)
        end
    end

    # M22: DsDsbar → DsDsbar
    @threads for i in 1:50
        @inbounds for j in 1:50
            if is_pcm(i) || is_pcm(j); continue; end
            Lc = Lcode_from_ij(i, j)
            p  = LSx2[i]
            p1 = LSx2[j]
            M22[i,j] = PartialWave(V_OBE_Ds, p, p1, J, Lc, Nnode) * Cutf(p, p1, lam)
        end
    end

    M21 = transpose(M12)
    M = [M11  M12;
         M21  M22]
    return M
end

Vpart_DD = OTBEV_DD(900.0)

function Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1, pcm2)

    @inline is_pcm(i) = (i == 25 || i == 50)

    LSx1 = [LSx; pcm1]; LSx1 = vcat(LSx1, LSx1, LSx1, LSx1)
    LSx2 = [LSx; pcm2]; LSx2 = vcat(LSx2, LSx2, LSx2, LSx2)

    LEC1 = LEC[1:2, :]   
    LEC2 = LEC[3:4, :]   
    LEC3 = LEC[5:6, :]   

    V_OBE_DD = (p, p1, x) -> DDbarobe(p, p1, x, Iso, gD, alp, f0, mD)
    V_OBE_NL = (p, p1, x) -> DDbarDsDsobe(p, p1, x, Iso, gD, alp, f0, mD, mDs)
    V_OBE_Ds = (p, p1, x) -> DsDsbarobeDs(p, p1, x, Iso, gD, alp, f0, mDs)

    M11 = zeros(ComplexF64, 50, 50)
    M12 = zeros(ComplexF64, 50, 50)
    M22 = zeros(ComplexF64, 50, 50)

    @inline function Lcode_from_ij(i, j)
        a = (i-1) ÷ 25
        b = (j-1) ÷ 25
        return (a==0 && b==0) ? 4 :
               (a==0 && b==1) ? 6 :
               (a==1 && b==0) ? 5 : 3
    end

    # M11: DDbar → DDbar
    @threads for i in 1:50
        @inbounds for j in 1:50
            Lc = Lcode_from_ij(i, j)
            p  = LSx1[i]
            p1 = LSx1[j]
            obe = (is_pcm(i) || is_pcm(j)) ?
                    PartialWave(V_OBE_DD, p, p1, J, Lc, Nnode) : 0.0
            M11[i,j] = (obe + DDbarContact(p, p1, LEC1, Iso, J, Lc)) * Cutf(p, p1, lam)
        end
    end

    # M12: DDbar → DsDsbar
    @threads for i in 1:50
        @inbounds for j in 1:50
            Lc = Lcode_from_ij(i, j)
            p  = LSx1[i]
            p1 = LSx2[j]
            obe = (is_pcm(i) || is_pcm(j)) ?
                    PartialWave(V_OBE_NL, p, p1, J, Lc, Nnode) : 0.0
            M12[i,j] = (obe + DDbarDsDsContact(p, p1, LEC2, Iso, J, Lc)) * Cutf(p, p1, lam)
        end
    end

    # M22: DsDsbar → DsDsbar
    @threads for i in 1:50
        @inbounds for j in 1:50
            Lc = Lcode_from_ij(i, j)
            p  = LSx2[i]
            p1 = LSx2[j]
            obe = (is_pcm(i) || is_pcm(j)) ?
                    PartialWave(V_OBE_Ds, p, p1, J, Lc, Nnode) : 0.0
            M22[i,j] = (obe + DsDsbarContact(p, p1, LEC3, Iso, J, Lc)) * Cutf(p, p1, lam)
        end
    end

    M21 = transpose(M12)
    M1  = [M11  M12;
           M21  M22]
    M   = Vpart_DD + M1
    return M
end

function Propagator_D(Ecm, pon2, mB, LSx, LSw, Nnode)
    N1  = Nnode + 1
    x2L = LSx .^ 2
    EL  = sqrt.(x2L .+ mB^2)
    ProList = LSw .* x2L .* (0.25*Ecm .+ 0.5*EL) ./ (pon2 .- x2L)
    if pon2 > 0
        P1     = 0.5 * LSw * Ecm * pon2 ./ (pon2 .- LSx .^ 2)
        pon    = sqrt(pon2)
        Proend = -sum(P1) - 1im*pi*0.25*pon*Ecm
    else
        Proend = 0.0
    end
    ProList = [ProList; Proend] / (2*pi)^3
    return ProList
end

function GMatrix_DD(Ecm, pon21, pon22)
    G1 = Propagator_D(Ecm, pon21, mD,  LSx, LSw, Nnode)
    G2 = Propagator_D(Ecm, pon22, mDs, LSx, LSw, Nnode)
    G1 = LinearAlgebra.Diagonal(G1)
    G2 = LinearAlgebra.Diagonal(G2)
    Z  = zeros(eltype(G1), 25, 25)

    GM1 = [G1  Z;  Z  G1]   # DDbar 传播子
    GM2 = [G2  Z;  Z  G2]   # DsDsbar 传播子
    Z1  = zeros(eltype(GM1), 50, 50)

    GM = [GM1  Z1;
          Z1   GM2]
    return GM
end

function Tmatrix_D(VM, Gmtr)
    T = (I - VM * Gmtr) \ VM
    return T
end


function fD0(k, GmD, GeD)
    s_eff = GmD + mD / (2*sqrt(mD^2 + k^2)) * GeD
    return s_eff * Cutborn(k)
end

function fD2(k, GmD, GeD)
    s_eff = 1/sqrt(2) * (GmD - mD / sqrt(mD^2 + k^2) * GeD)
    return s_eff * Cutborn(k)
end

function fDs0(k, GmDs, GeDs)
    s_eff = GmDs + mDs / (2*sqrt(mDs^2 + k^2)) * GeDs
    return s_eff * Cutborn(k)
end

function fDs2(k, GmDs, GeDs)
    s_eff = 1/sqrt(2) * (GmDs - mDs / sqrt(mDs^2 + k^2) * GeDs)
    return s_eff * Cutborn(k)
end

function fD0m(L, pcm1, GmD, GeD)
    fm   = zeros(ComplexF64, 1, 25)
    lsx1 = [LSx; pcm1]; lsx1 = vcat(lsx1, lsx1, lsx1, lsx1)
    for i in 1:25
        fm[1, i] = (L == 0) ? fD0(lsx1[i], GmD, GeD) :
                               fD2(lsx1[i], GmD, GeD)
    end
    return fm
end

function fDs0m(L, pcm2, GmDs, GeDs)
    fm   = zeros(ComplexF64, 1, 25)
    lsx2 = [LSx; pcm2]; lsx2 = vcat(lsx2, lsx2, lsx2, lsx2)
    for i in 1:25
        fm[1, i] = (L == 0) ? fDs0(lsx2[i], GmDs, GeDs) :
                               fDs2(lsx2[i], GmDs, GeDs)
    end
    return fm
end

function fDD(Ecm, G1, G2, TM, GmD, GeD, GmDs, GeDs)
    pcm1 = sqrt((Ecm/2)^2 - mD^2)
    pcm2 = sqrt((Ecm/2)^2 - mDs^2)
    T_DD_00  = TM[1:25,   75]   
    T_DD_02  = TM[1:25,  100]   
    T_DD_20  = TM[26:50,  75]   
    T_DD_22  = TM[26:50, 100]

    T_Ds_00  = TM[51:75,  75]   
    T_Ds_02  = TM[51:75, 100]
    T_Ds_20  = TM[76:100, 75]
    T_Ds_22  = TM[76:100,100]

    fll0 = fD0(pcm1, GmD, GeD) +
           only(fD0m(0, pcm1, GmD, GeD)  * G1 * T_DD_00  +
                fD0m(2, pcm1, GmD, GeD)  * G1 * T_DD_20  +
                fDs0m(0, pcm2, GmDs, GeDs) * G2 * T_Ds_00 +
                fDs0m(2, pcm2, GmDs, GeDs) * G2 * T_Ds_20)

    fll2 = fD2(pcm1, GmD, GeD) +
           only(fD0m(0, pcm1, GmD, GeD)  * G1 * T_DD_02  +
                fD0m(2, pcm1, GmD, GeD)  * G1 * T_DD_22  +
                fDs0m(0, pcm2, GmDs, GeDs) * G2 * T_Ds_02 +
                fDs0m(2, pcm2, GmDs, GeDs) * G2 * T_Ds_22)

    return fll0, fll2
end

function solveGME_D(Ecm, f0_, f2_)
    s    = Ecm^2
    gm   = 2/3 * (f0_ + 1/sqrt(2) * f2_)
    ge   = (f0_/sqrt(2) - f2_) * sqrt(2*s) / (3*mD)
    absGM = abs(gm)
    absGE = abs(ge)
    r     = abs(ge / gm)
    return absGM, absGE, r
end

function cross_DD(Ecm, lam, para)
    GmD  = para[6] + para[7]*im;  GeD  = GmD
    GmDs = para[8] + para[9]*im;  GeDs = GmDs

    LEC    = zeros(6, 9)
    LEC[1,6] = para[1];  LEC[3,6] = para[2];  LEC[5,6] = para[3]
    LECann = zeros(2, 10)
    LECann[1,6] = para[4];  LECann[2,6] = para[5]

    pcm1 = sqrt((Ecm/2)^2 - mD^2)
    pcm2 = sqrt((Ecm/2)^2 - mDs^2)
    kcm  = sqrt((Ecm/2)^2 - mee^2)

    pon21 = pcm1^2
    pon22 = pcm2^2
    s     = Ecm^2

    fee0 = 1.0 + mee/Ecm              
    fee2 = 1/sqrt(2)*(1 - 2*mee/Ecm)  

    fDD_1 = pcm1 / sqrt(s)   

    ccc = -4/9 * Alpha   

    beta = pcm1 / kcm    

    VM   = Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1, pcm2)
    Gmtr = GMatrix_DD(Ecm, pon21, pon22)
    TM   = Tmatrix_D(VM, Gmtr)

    G1   = Propagator_D(Ecm, pon21, mD,  LSx, LSw, Nnode)
    G1   = LinearAlgebra.Diagonal(G1)
    G2   = Propagator_D(Ecm, pon22, mDs, LSx, LSw, Nnode)
    G2   = LinearAlgebra.Diagonal(G2)

    fall0, fall2 = fDD(Ecm, G1, G2, TM, GmD, GeD, GmDs, GeDs)

    F00 = ccc * fall0 * fee0
    F02 = ccc * fall0 * fee2
    F20 = ccc * fall2 * fee0
    F22 = ccc * fall2 * fee2

    sigam = 3*pi*beta/s * cs * (abs2(F00) + abs2(F02) + abs2(F20) + abs2(F22)) * hc^2 * 1e10
    return sigam
end

function Geff_DD(Ecm, lam, para)
    s       = Ecm^2
    crosscc = cross_DD(Ecm, lam, para) / (hc^2 * 1e10)
    pcm1    = sqrt((Ecm/2)^2 - mD^2)
    kcm     = sqrt((Ecm/2)^2 - mee^2)
    beta    = pcm1 / kcm
    denom   = 4*pi*Alpha^2*beta / (3*s) * cs * (1 + 2*mD^2/s)
    return sqrt(crosscc / denom)
end

function ll_DD(Ecm, lam, para)
    GmD  = para[6] + para[7]*im;  GeD  = GmD
    GmDs = para[8] + para[9]*im;  GeDs = GmDs

    LEC    = zeros(6, 9)
    LEC[1,6] = para[1];  LEC[3,6] = para[2];  LEC[5,6] = para[3]
    LECann = zeros(2, 10)
    LECann[1,6] = para[4];  LECann[2,6] = para[5]

    pcm1  = sqrt((Ecm/2)^2 - mD^2)
    pcm2  = sqrt((Ecm/2)^2 - mDs^2)
    pon21 = pcm1^2
    pon22 = pcm2^2

    VM   = Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1, pcm2)
    Gmtr = GMatrix_DD(Ecm, pon21, pon22)
    TM   = Tmatrix_D(VM, Gmtr)

    G1 = LinearAlgebra.Diagonal(Propagator_D(Ecm, pon21, mD,  LSx, LSw, Nnode))
    G2 = LinearAlgebra.Diagonal(Propagator_D(Ecm, pon22, mDs, LSx, LSw, Nnode))

    fall0, fall2 = fDD(Ecm, G1, G2, TM, GmD, GeD, GmDs, GeDs)
    gm1, ge1, r1 = solveGME_D(Ecm, fall0, fall2)
    return gm1, ge1, r1
end

function SMatrix_DD(TM, Ecm)
    E    = Ecm / 2
    pcm1 = sqrt(E^2 - mD^2)
    T1 = TM[25, 25];  T2 = TM[25,  50]
    T3 = TM[50, 25];  T4 = TM[50,  50]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    TT = [T1 T2; T3 T4]
    S  = I2 - im/(8*pi^2) * pcm1 * E * TT
    return S
end

function dcross_DD(Ecm, lam, para, x)
    xs   = sqrt(1 - x^2)
    s    = Ecm^2
    pcm1 = sqrt((Ecm/2)^2 - mD^2)
    kcm  = sqrt((Ecm/2)^2 - mee^2)
    beta = pcm1 / kcm

    GmD, GeD = para[6]+para[7]*im, para[6]+para[7]*im

    LEC    = zeros(6, 9)
    LEC[1,6] = para[1];  LEC[3,6] = para[2];  LEC[5,6] = para[3]
    LECann = zeros(2, 10)
    LECann[1,6] = para[4];  LECann[2,6] = para[5]

    pcm2  = sqrt((Ecm/2)^2 - mDs^2)
    GmDs  = para[8]+para[9]*im;  GeDs = GmDs
    pon21 = pcm1^2;  pon22 = pcm2^2

    VM   = Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1, pcm2)
    Gmtr = GMatrix_DD(Ecm, pon21, pon22)
    TM   = Tmatrix_D(VM, Gmtr)
    G1   = LinearAlgebra.Diagonal(Propagator_D(Ecm, pon21, mD,  LSx, LSw, Nnode))
    G2   = LinearAlgebra.Diagonal(Propagator_D(Ecm, pon22, mDs, LSx, LSw, Nnode))

    fall0, fall2 = fDD(Ecm, G1, G2, TM, GmD, GeD, GmDs, GeDs)
    xi = Alpha^2 * beta / (4*s)
    dc = xi * (fall0^2 * (1 + x^2) + 4*mD^2/s * fall2^2 * xs^2) * hc^2 * 1e10 * 2*pi
    return dc
end

function scan_cross_DD(Ecm_range, lam, para)
    return [cross_DD(Ecm, lam, para) for Ecm in Ecm_range]
end

function DDbarT(Ecm, lam, para)
    GmD  = para[6]+para[7]*im
    GmDs = para[8]+para[9]*im
    LEC    = zeros(6, 9)
    LEC[1,6] = para[1];  LEC[3,6] = para[2];  LEC[5,6] = para[3]
    LECann = zeros(2, 10)
    LECann[1,6] = para[4];  LECann[2,6] = para[5]

    pcm1 = sqrt((Ecm/2)^2 - mD^2)
    pcm2 = sqrt((Ecm/2)^2 - mDs^2)
    pon21 = pcm1^2;  pon22 = pcm2^2

    VM   = Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1, pcm2)
    Gmtr = GMatrix_DD(Ecm, pon21, pon22)
    TM   = Tmatrix_D(VM, Gmtr)

    qu = sqrt((Ecm/2)^2 - mD^2)
    qv = sqrt((Ecm/2)^2 - mDs^2)

    a00 = -pi*Ecm/4 * TM[25, 75] * sqrt(2) * sqrt(qu*qv)
    a02 = -pi*Ecm/4 * TM[25,100] * sqrt(2) * sqrt(qu*qv)
    a20 = -pi*Ecm/4 * TM[50, 75] * sqrt(2) * sqrt(qu*qv)
    a22 = -pi*Ecm/4 * TM[50,100] * sqrt(2) * sqrt(qu*qv)

    return real(a00), imag(a00), real(a02), imag(a02),
           real(a20), imag(a20), real(a22), imag(a22)
end