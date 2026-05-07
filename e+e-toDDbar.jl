using Base.Threads
using LinearAlgebra
using Printf, Plots
using QuadGK
include("General.jl")

const mD  = 1864.84   # D 介子质量
const mDs = 1968.34   # Ds 介子质量（DsDsbar 通道暂时保留为注释，不参与计算）
const mee = 0.511

const Alpha = 1.0/137.036   # 精细结构常数
const hc    = 197.3269804   # MeV·fm
const cs    = 3.893793656e5

# 默认的 DDbar 单通道参数；调用函数时也可以通过参数显式覆盖。
const Nnode = 24            # LSpoints 返回 24 个积分节点，外加 1 个在壳动量点 = 25
const J     = 1             # 总角动量（对于 1P1，J=1）
const Iso   = 0             # 同位旋（0 或 1）
const f0    = 92.4          # 介子衰变常数（MeV）
const gD    = 0.57          # D*Dπ 耦合常数
const alp   = 1.0           # 截断参数
const LS_POINTS = LSpoints(Nnode, 700.0)
const LSx = LS_POINTS[1]
const LSw = LS_POINTS[2]

const DD_CHANNEL_SIZE = Nnode + 1
const DD_MATRIX_SIZE  = 2 * DD_CHANNEL_SIZE

function Cutf(p, p1, lam)
    return exp(-(p^6 + p1^6) / lam^6)
end

function Cutborn(k, lam_born=900.0)
    return exp(-(k^6) / lam_born^6)
end

function qsq(p, p1, x)
    return p^2 + p1^2 - 2*p*p1*x
end

function legendreP(l, x)
    l == 0 && return one(x)
    l == 1 && return x
    p0 = one(x)
    p1 = x
    for n in 2:l
        p0, p1 = p1, ((2*n - 1) * x * p1 - (n - 1) * p0) / n
    end
    return p1
end

function partial_wave_weight(Lc, x)
    if Lc == 4              # S -> S
        return 1.0
    elseif Lc == 6 || Lc == 5  # S <-> D
        return legendreP(2, x)
    elseif Lc == 3          # D -> D
        return legendreP(2, x)^2
    else
        error("unsupported partial-wave code: $Lc")
    end
end

function PartialWave(V, p, p1, J, Lc, nnode)
    xnodes, wnodes = Gausspoints(nnode, -1.0, 1.0)
    val = zero(ComplexF64)
    @inbounds for k in eachindex(xnodes)
        x = xnodes[k]
        val += wnodes[k] * partial_wave_weight(Lc, x) * V(p, p1, x)
    end
    return val / 2
end

function DDbarobe(p, p1, x, Iso, gD, alp, f0, mD_)
    q = sqrt(max(qsq(p, p1, x), 0.0))
    return -Iso * gD^2 / (f0^2) * q^2 / (q^2 + alp^2 + eps())
end

function DDbartbe(q, Iso, gD, alp, f0)
    return Iso * gD^4 * FootballC_unequal(q, mD, mD, f0)
end

function DDbarContact(p, p1, LEC, Iso, J, Lc)
    if Lc == 4
        return LEC[1, 6]
    elseif Lc == 3
        return LEC[2, 6] * p^2 * p1^2
    elseif Lc == 5 || Lc == 6
        return 0.5 * (LEC[1, 6] + LEC[2, 6]) * p * p1
    else
        return 0.0
    end
end

function channel_momenta(pcm)
    nodes = [LSx; pcm]
    return vcat(nodes, nodes)
end

@inline function is_pcm_index(i)
    return i == DD_CHANNEL_SIZE || i == DD_MATRIX_SIZE
end

@inline function Lcode_from_ij(i, j)
    a = (i - 1) ÷ DD_CHANNEL_SIZE
    b = (j - 1) ÷ DD_CHANNEL_SIZE
    return (a == 0 && b == 0) ? 4 :
           (a == 0 && b == 1) ? 6 :
           (a == 1 && b == 0) ? 5 : 3
end

function OTBEV_DD(lam; iso=Iso, gd=gD, cutoff_alpha=alp, decay_const=f0, jtot=J)
    pgrid = channel_momenta(0.0)
    V_OBE_DD = (p, p1, x) -> DDbarobe(p, p1, x, iso, gd, cutoff_alpha, decay_const, mD)
    # V_TBE_DD 可在需要时重新加入；当前 DDbar 单道流程保持与原始代码一致，仅使用 OBE。

    M11 = zeros(ComplexF64, DD_MATRIX_SIZE, DD_MATRIX_SIZE)   # DDbar → DDbar

    @threads for i in 1:DD_MATRIX_SIZE
        @inbounds for j in 1:DD_MATRIX_SIZE
            if is_pcm_index(i) || is_pcm_index(j)
                continue
            end
            Lc = Lcode_from_ij(i, j)
            p  = pgrid[i]
            p1 = pgrid[j]
            obe = PartialWave(V_OBE_DD, p, p1, jtot, Lc, Nnode)
            M11[i, j] = obe * Cutf(p, p1, lam)
        end
    end

    #=
    # DsDsbar 耦合通道暂时注释掉：
    # M12: DDbar → DsDsbar
    # M22: DsDsbar → DsDsbar
    # M21 = transpose(M12)
    # M = [M11 M12; M21 M22]
    =#

    return M11
end

const Vpart_DD = OTBEV_DD(900.0)

function Vmatrixss_DD(Iso_, gD_, alp_, f0_, mD_, mDs_, J_, LEC, LECann, lam, pcm1, pcm2=nothing)
    pgrid = channel_momenta(pcm1)
    LEC1 = LEC[1:2, :]

    V_OBE_DD = (p, p1, x) -> DDbarobe(p, p1, x, Iso_, gD_, alp_, f0_, mD_)

    M11 = zeros(ComplexF64, DD_MATRIX_SIZE, DD_MATRIX_SIZE)

    @threads for i in 1:DD_MATRIX_SIZE
        @inbounds for j in 1:DD_MATRIX_SIZE
            Lc = Lcode_from_ij(i, j)
            p  = pgrid[i]
            p1 = pgrid[j]
            obe = (is_pcm_index(i) || is_pcm_index(j)) ?
                  PartialWave(V_OBE_DD, p, p1, J_, Lc, Nnode) : 0.0
            M11[i, j] = (obe + DDbarContact(p, p1, LEC1, Iso_, J_, Lc)) * Cutf(p, p1, lam)
        end
    end

    #=
    # DsDsbar 耦合通道暂时注释掉：
    # LEC2 = LEC[3:4, :]
    # LEC3 = LEC[5:6, :]
    # M12 = zeros(ComplexF64, DD_MATRIX_SIZE, DD_MATRIX_SIZE)
    # M22 = zeros(ComplexF64, DD_MATRIX_SIZE, DD_MATRIX_SIZE)
    # M21 = transpose(M12)
    # M = [M11 M12; M21 M22]
    =#

    return Vpart_DD + M11
end

function Propagator_D(Ecm, pon2, mB, LSx_, LSw_, nnode)
    x2L = LSx_ .^ 2
    EL  = sqrt.(x2L .+ mB^2)
    ProList = LSw_ .* x2L .* (0.25*Ecm .+ 0.5*EL) ./ (pon2 .- x2L)
    if pon2 > 0
        P1     = 0.5 * LSw_ * Ecm * pon2 ./ (pon2 .- LSx_ .^ 2)
        pon    = sqrt(pon2)
        Proend = -sum(P1) - 1im*pi*0.25*pon*Ecm
    else
        Proend = 0.0
    end
    ProList = [ProList; Proend] / (2*pi)^3
    return ProList
end

function GMatrix_DD(Ecm, pon21, pon22=nothing)
    G1 = Propagator_D(Ecm, pon21, mD, LSx, LSw, Nnode)
    G1 = LinearAlgebra.Diagonal(G1)
    Z  = zeros(eltype(G1), DD_CHANNEL_SIZE, DD_CHANNEL_SIZE)
    GM = [G1 Z; Z G1]   # DDbar 单通道传播子（S/D 两个分波）

    #=
    # DsDsbar 传播子暂时注释掉：
    # G2 = Propagator_D(Ecm, pon22, mDs, LSx, LSw, Nnode)
    # GM2 = [G2 Z; Z G2]
    # GM = [GM1 Z1; Z1 GM2]
    =#

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

#=
# DsDsbar 源项暂时注释掉：
# function fDs0(k, GmDs, GeDs) ... end
# function fDs2(k, GmDs, GeDs) ... end
=#

function fD0m(L, pcm1, GmD, GeD)
    fm   = zeros(ComplexF64, 1, DD_CHANNEL_SIZE)
    lsx1 = [LSx; pcm1]
    for i in 1:DD_CHANNEL_SIZE
        fm[1, i] = (L == 0) ? fD0(lsx1[i], GmD, GeD) : fD2(lsx1[i], GmD, GeD)
    end
    return fm
end

function fDD(Ecm, G1, TM, GmD, GeD)
    pcm1 = sqrt((Ecm/2)^2 - mD^2)
    T_DD_00 = TM[1:DD_CHANNEL_SIZE, DD_CHANNEL_SIZE]
    T_DD_02 = TM[1:DD_CHANNEL_SIZE, DD_MATRIX_SIZE]
    T_DD_20 = TM[(DD_CHANNEL_SIZE+1):DD_MATRIX_SIZE, DD_CHANNEL_SIZE]
    T_DD_22 = TM[(DD_CHANNEL_SIZE+1):DD_MATRIX_SIZE, DD_MATRIX_SIZE]

    fll0 = fD0(pcm1, GmD, GeD) +
           only(fD0m(0, pcm1, GmD, GeD) * G1 * T_DD_00 +
                fD0m(2, pcm1, GmD, GeD) * G1 * T_DD_20)

    fll2 = fD2(pcm1, GmD, GeD) +
           only(fD0m(0, pcm1, GmD, GeD) * G1 * T_DD_02 +
                fD0m(2, pcm1, GmD, GeD) * G1 * T_DD_22)

    #=
    # DsDsbar 对 DDbar 末态的贡献暂时注释掉：
    # + fDs0m(...) * G2 * T_Ds_..
    =#

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

function dd_lecs(para)
    LEC = zeros(2, 9)
    LEC[1, 6] = para[1]
    LEC[2, 6] = para[2]
    return LEC
end

function cross_DD(Ecm, lam, para)
    GmD = para[3] + para[4]*im
    GeD = GmD

    LEC    = dd_lecs(para)
    LECann = zeros(2, 10)

    pcm1  = sqrt((Ecm/2)^2 - mD^2)
    kcm   = sqrt((Ecm/2)^2 - mee^2)
    pon21 = pcm1^2
    s     = Ecm^2

    fee0 = 1.0 + mee/Ecm
    fee2 = 1/sqrt(2)*(1 - 2*mee/Ecm)

    ccc  = -4/9 * Alpha
    beta = pcm1 / kcm

    VM   = Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1)
    Gmtr = GMatrix_DD(Ecm, pon21)
    TM   = Tmatrix_D(VM, Gmtr)

    G1 = LinearAlgebra.Diagonal(Propagator_D(Ecm, pon21, mD, LSx, LSw, Nnode))

    fall0, fall2 = fDD(Ecm, G1, TM, GmD, GeD)

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
    GmD = para[3] + para[4]*im
    GeD = GmD

    LEC    = dd_lecs(para)
    LECann = zeros(2, 10)

    pcm1  = sqrt((Ecm/2)^2 - mD^2)
    pon21 = pcm1^2

    VM   = Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1)
    Gmtr = GMatrix_DD(Ecm, pon21)
    TM   = Tmatrix_D(VM, Gmtr)

    G1 = LinearAlgebra.Diagonal(Propagator_D(Ecm, pon21, mD, LSx, LSw, Nnode))

    fall0, fall2 = fDD(Ecm, G1, TM, GmD, GeD)
    gm1, ge1, r1 = solveGME_D(Ecm, fall0, fall2)
    return gm1, ge1, r1
end

function SMatrix_DD(TM, Ecm)
    E    = Ecm / 2
    pcm1 = sqrt(E^2 - mD^2)
    T1 = TM[DD_CHANNEL_SIZE, DD_CHANNEL_SIZE]
    T2 = TM[DD_CHANNEL_SIZE, DD_MATRIX_SIZE]
    T3 = TM[DD_MATRIX_SIZE, DD_CHANNEL_SIZE]
    T4 = TM[DD_MATRIX_SIZE, DD_MATRIX_SIZE]
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

    GmD, GeD = para[3]+para[4]*im, para[3]+para[4]*im

    LEC    = dd_lecs(para)
    LECann = zeros(2, 10)
    pon21  = pcm1^2

    VM   = Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1)
    Gmtr = GMatrix_DD(Ecm, pon21)
    TM   = Tmatrix_D(VM, Gmtr)
    G1   = LinearAlgebra.Diagonal(Propagator_D(Ecm, pon21, mD, LSx, LSw, Nnode))

    fall0, fall2 = fDD(Ecm, G1, TM, GmD, GeD)
    xi = Alpha^2 * beta / (4*s)
    dc = xi * (abs2(fall0) * (1 + x^2) + 4*mD^2/s * abs2(fall2) * xs^2) * hc^2 * 1e10 * 2*pi
    return dc
end

function scan_cross_DD(Ecm_range, lam, para)
    return [cross_DD(Ecm, lam, para) for Ecm in Ecm_range]
end

function DDbarT(Ecm, lam, para)
    LEC    = dd_lecs(para)
    LECann = zeros(2, 10)

    pcm1  = sqrt((Ecm/2)^2 - mD^2)
    pon21 = pcm1^2

    VM   = Vmatrixss_DD(Iso, gD, alp, f0, mD, mDs, J, LEC, LECann, lam, pcm1)
    Gmtr = GMatrix_DD(Ecm, pon21)
    TM   = Tmatrix_D(VM, Gmtr)

    qu = sqrt((Ecm/2)^2 - mD^2)

    a00 = -pi*Ecm/4 * TM[DD_CHANNEL_SIZE, DD_CHANNEL_SIZE] * qu
    a02 = -pi*Ecm/4 * TM[DD_CHANNEL_SIZE, DD_MATRIX_SIZE] * qu
    a20 = -pi*Ecm/4 * TM[DD_MATRIX_SIZE, DD_CHANNEL_SIZE] * qu
    a22 = -pi*Ecm/4 * TM[DD_MATRIX_SIZE, DD_MATRIX_SIZE] * qu

    #=
    # DsDsbar 跃迁振幅暂时注释掉：
    # qv = sqrt((Ecm/2)^2 - mDs^2)
    # aDD_to_DsDs = -pi*Ecm/4 * TM[..., ...] * sqrt(2) * sqrt(qu*qv)
    =#

    return real(a00), imag(a00), real(a02), imag(a02),
           real(a20), imag(a20), real(a22), imag(a22)
end
