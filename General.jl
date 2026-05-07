include("NumIntegrate.jl")
function wf(q, m1, m2)
    m12p = (m1 + m2)^2
    m12m = (m1 - m2)^2
    return sqrt(q^2 + m12p) * sqrt(q^2 + m12m) / q
end

function Lf(q, m1, m2)
    w     = wf(q, m1, m2)
    m12m  = (m1^2 - m2^2)^2
    num   = (q * w + q^2)^2 - m12m
    den   = 4 * m1 * m2 * q^2
    return w * log(num / den) / 2 / q
end


function AB_integrals(q, m1, m2)
    m12  = m1^2;   m22  = m2^2
    mm2p = m12 + m22
    mm2m = m12 - m22
    mm4p = m12^2 + m22^2
    mm4m = m12^2 - m22^2
    logm = (m1 != m2) ? log(m1 / m2) : 0.0
    w    = wf(q, m1, m2)
    L    = Lf(q, m1, m2)

    A = (1 / (24*pi^2)) * (
          - (mm2m^2 + 4*q^2*mm2p + 5*q^4/3) / q^2
          - mm2m / q^4 * ((mm2m^2) + 3*q^2*mm2p) * logm
          + 2 * w^2 * L
        )

    B = -(1 / (24*pi^2)) * (
          - (5*q^4/3 + 4*q^2*mm2p + 4*mm2m^2) / q^4
          - mm2m / q^6 * (4*mm2m^2 + 6*q^2*mm2p + 3*q^4) * logm
          + 2*L / q^4 * (q^4 + 2*q^2*mm2p + 4*mm2m^2)
        )

    return A, B
end

#  1. 足球图（Football diagram）
function FootballC_unequal(qMeV, m1MeV, m2MeV, f0MeV)
    q  = qMeV  / 1000.0
    m1 = m1MeV / 1000.0
    m2 = m2MeV / 1000.0
    f0 = f0MeV / 1000.0

    C0 = 1.0 / (3072 * pi^2 * f0^4)

    if abs(q) < 0.02
        m12  = m1^2;  m22  = m2^2
        mm2p = m12 + m22;  mm2m = m12 - m22
        mm   = m12 * m22
        m14  = m12^2;  m24  = m22^2
        m16  = m12^3;  m26  = m22^3
        m18  = m14^2;  m28  = m24^2
        mm4p = m14 + m24;  mm4m = m14 - m24
        mm6p = m16 + m26;  mm6m = m16 - m26
        mm8p = m18 + m28;  mm8m = m18 - m28
        logm = (m1 != m2) ? log(m1 / m2) : 0.0

        # V0 = V00 + q²V02 + q⁴V04 + q⁶V06
        V00 = 1.25 * mm2p + 1.5 * mm4p * logm / mm2m
        V02 = (11*mm6m - 21*mm*mm2m + 6*mm2p*(mm4p - 4*mm)*logm) /
              (12 * mm2m^3)
        V04 = (mm8m - 8*mm*mm4m + 24*mm^2*logm) / (8 * mm2m^5)
        V06 = -(mm2m*(mm8p - 14*mm*mm4p - 94*mm^2) +
                120*mm^2*mm2p*logm) / (40 * mm2m^7)
        V0  = V00 + q^2*V02 + q^4*V04 + q^6*V06
    else
        L    = Lf(q, m1, m2)
        w    = wf(q, m1, m2)
        m12  = m1^2;  m22  = m2^2
        mm2p = m12 + m22;  mm2m = m12 - m22
        logm = (m1 != m2) ? log(m1 / m2) : 0.0

        V0 = -0.5 * mm2m^2 / q^2 \
             - 0.5 * mm2m * (mm2m^2 + 3*mm2p*q^2) * logm / q^4 \
             + w^2 * L
    end

    return C0 * V0 / 10^6
end

#  2. 三角图（Triangle diagram）
function hyp2F1_half1c(c, z; Nmax=200)
    # ₂F₁(1/2,1;c;z) = Σ_{n=0}^∞ (1/2)_n (1)_n / (c)_n / n! * z^n
    term = one(z)
    s    = term
    a1   = 0.5   
    a2   = 1.0   
    cn   = c     
    for n in 1:Nmax
        term *= (a1 * a2) / (cn * n) * z
        s    += term
        a1   += 1.0
        a2   += 1.0
        cn   += 1.0
        abs(term) < 1e-15 * abs(s) && break
    end
    return s
end

function F1_func(z)
    if abs(z) < 1e-10
        return 0.0 + 0.0im
    end
    z  = z + 0im    
    h  = hyp2F1_half1c(2.0, z)
    t1 = 3 * z * h
    sq = sqrt(1 - z)
    t2 = 6 * (log(z) + sq * (log(4/z - 4 + 0im) - 1))
    return t1 + t2
end

function F2_func(z)
    if abs(z) < 1e-10
        return 0.0 + 0.0im
    end
    z  = z + 0im
    sq = sqrt(1 - z)
    # ₂F₁(1/2,1;1;z) = 1/√(1-z)
    t1 = 1 / sq
    t2 = log(4/z - 4 + 0im) / sq
    return t1 - t2
end

# M²(x) = 4x(1-x)q² + 4(m1²-m2²)x + 4m2²
function Msq(x, q, m1, m2)
    return 4*x*(1-x)*q^2 + 4*(m1^2 - m2^2)*x + 4*m2^2
end

function gauss_integrate(f, a, b; N=128)
    nodes, weights = gauss_legendre_nodes_weights(N)
    mid  = (a + b) / 2
    half = (b - a) / 2
    s    = zero(f((a+b)/2))
    for i in 1:N
        s += weights[i] * f(mid + half * nodes[i])
    end
    return half * s
end

function gauss_legendre_nodes_weights(N)
    x = range(-1+1/N, 1-1/N, length=N) |> collect
    w = fill(2.0/N, N)
    return x, w
end

function TriangleC_unequal(qMeV, m1MeV, m2MeV, DeltaMeV, f0MeV)
    q   = qMeV    / 1000.0
    m1  = m1MeV   / 1000.0
    m2  = m2MeV   / 1000.0
    Δ   = DeltaMeV/ 1000.0
    f0  = f0MeV   / 1000.0

    C0 = -1.0 / (768 * pi^2 * f0^2)

    if abs(Δ) < 1e-8
        # ∆→0
        if abs(q) < 0.01 && m1 == m2
            m12 = m1^2
            V0  = 16*m12 + 34/3*q^2 + 0.7*q^4/m12 - 9/140*q^6/m12^2
        elseif abs(q) < 0.01
            m12  = m1^2;  m22  = m2^2
            mm2p = m12 + m22;  mm2m = m12 - m22
            mm   = m12 * m22
            m14  = m12^2;  m24  = m22^2
            m16  = m12^3;  m26  = m22^3
            m18  = m14^2;  m28  = m24^2
            mm4p = m14 + m24;  mm4m = m14 - m24
            mm6m = m16 - m26;  mm8m = m18 - m28
            logm = (m1 != m2) ? log(m1/m2) : 0.0
            V00  = 3.5*mm2p + 9*mm4p*logm/mm2m
            V02  = (43*mm6m - 69*mm*mm2m +
                    30*mm2p*(mm4p - 4*mm)*logm) / (6*mm2m^3)
            V04  = 1.75*(mm8m - 8*mm*(m14-m24) + 24*mm^2*logm) / mm2m^5
            V06  = -0.45*(mm2m*(m18+m28 - 14*mm*(m14+m24) - 94*mm^2) +
                          120*mm^2*mm2p*logm) / mm2m^7
            V0   = V00 + q^2*V02 + q^4*V04 + q^6*V06
        else
            L    = Lf(q, m1, m2)
            m12  = m1^2;  m22  = m2^2
            mm2p = m12 + m22;  mm2m = m12 - m22
            logm = (m1 != m2) ? log(m1/m2) : 0.0
            V0   = mm2m^2/q^2 + (8*mm2p - 2*mm2m^2/q^2 + 10*q^2)*L +
                   mm2m*(mm2m^2 - 3*mm2p*q^2)*logm/q^4
        end
        return C0 * V0
    end

    # V_Triangle = -∆²/(4π²)*[4 + ∫₀¹ F₁(z)] - q²/(2π²)*∫₀¹ x(1-x)F₂(z) dx

    integrand_F1 = x -> begin
        Mq2 = Msq(x, q, m1, m2)
        z   = Mq2 / (4 * Δ^2)
        real(F1_func(z))
    end

    integrand_F2 = x -> begin
        Mq2 = Msq(x, q, m1, m2)
        z   = Mq2 / (4 * Δ^2)
        x * (1 - x) * real(F2_func(z))
    end

    int_F1 = gauss_integrate(integrand_F1, 0.0, 1.0; N=100)
    int_F2 = gauss_integrate(integrand_F2, 0.0, 1.0; N=100)

    V_tri = -Δ^2 / (4*pi^2) * (4.0 + int_F1) -
             q^2  / (2*pi^2) * int_F2

    return C0 * V_tri
end

function LSpoints(Nnode,C)
    x,w=Gausspoints(Nnode,-1,1);
    LSx=@. C*tan(0.25*(x+1)*pi);
    LSw=@. 0.25*C*pi*w/cos(0.25*(x+1)*pi)^2;
    return LSx,LSw
end

#  3. 平行箱图（Planar Box diagram）
function F3_func(z)
    if abs(z) < 1e-10
        return 0.0 + 0.0im
    end
    z  = z + 0im
    h  = hyp2F1_half1c(3.0, z)
    sq = sqrt(1 - z)
    t1 = 45 * z^2 * h
    t2 = 60 * (3*z - 2) * log(z)
    t3 = 140 * sq^3
    t4 = -120 * sq^3 * log(4/z - 4 + 0im)
    return t1 + t2 + t3 + t4
end

function dFdz(F_func, z; h=1e-6)
    return (F_func(z + h) - F_func(z - h)) / (2*h)
end

function BoxPlanarC_unequal(qMeV, m1MeV, m2MeV, DeltaMeV)
    q  = qMeV    / 1000.0
    m1 = m1MeV   / 1000.0
    m2 = m2MeV   / 1000.0
    Δ  = DeltaMeV/ 1000.0

    C0 = 1.0 / (192 * pi^2)    

    if abs(q) < 0.01 && m1 == m2
        m12 = m1^2
        V0 = 64*m12 + 136/3*q^2 + 4.3*q^4/m12 - 71/140*q^6/m12^2
    elseif abs(q) < 0.01
        m12  = m1^2;  m22  = m2^2
        mm2p = m12 + m22;  mm2m = m12 - m22
        mm   = m12 * m22
        m14  = m12^2;  m24  = m22^2
        m16  = m12^3;  m26  = m22^3
        m18  = m14^2;  m28  = m24^2
        mm4p = m14 + m24;  mm4m = m14 - m24
        mm6m = m16 - m26;  mm8m = m18 - m28
        mm8p = m18 + m28
        logm = (m1 != m2) ? log(m1/m2) : 0.0
        V00  = 9.5*mm2p + (33*mm4p + 24*mm)*logm/mm2m
        V02  = (169*mm6m - 87*mm*mm2m +
                6*mm2p*(23*mm4p - 116*mm)*logm) / (6*mm2m^3)
        V04  = 0.75*(13*mm8m - 152*mm*mm4m +
                     8*(4*mm4p*mm + 55*mm^2)*logm) / mm2m^5
        V06  = -0.05*(mm2m*(59*mm8p - 1706*mm*mm4p - 8586*mm^2) +
                      120*mm*mm2p*(4*mm4p + 91*mm)*logm) / mm2m^7
        V0   = V00 + q^2*V02 + q^4*V04 + q^6*V06
    else
        w    = wf(q, m1, m2)
        L    = Lf(q, m1, m2)
        m12  = m1^2;  m22  = m2^2
        mm2p = m12 + m22;  mm2m = m12 - m22
        logm = (m1 != m2) ? log(m1/m2) : 0.0
        mm   = m12 * m22
        mm4p = m12^2 + m22^2
        V00  = mm2m^2/q^2 + mm2m*(mm2m^2 - 9*q^2*mm2p)*logm/q^4
        V01  = 2/w^2*(23*q^4 - mm2m^4/q^4 + 56*mm2p*q^2 +
                      8*mm2p*mm2m^2/q^2 + 42*mm4p + 44*mm)*L
        V0   = V00 + V01
    end

    if abs(Δ) < 1e-8
        return V0 * C0 * 10^6
    end

    integrand1 = x -> begin
        Mq2 = Msq(x, q, m1, m2)
        z   = Mq2 / (4 * Δ^2)
        dF3 = real(dFdz(F3_func, z))
        102.0 + dF3
    end

    # 第二项（含 F₁'(z)）
    integrand2 = x -> begin
        Mq2 = Msq(x, q, m1, m2)
        z   = Mq2 / (4 * Δ^2)
        (-10*x^2 + 10*x - 1) * real(dFdz(F1_func, z))
    end

    # 第三项（含 F₂'(z)）
    integrand3 = x -> begin
        Mq2 = Msq(x, q, m1, m2)
        z   = Mq2 / (4 * Δ^2)
        x^2 * (1 - x)^2 * real(dFdz(F2_func, z))
    end

    int1 = gauss_integrate(integrand1, 0.0, 1.0; N=100)
    int2 = gauss_integrate(integrand2, 0.0, 1.0; N=100)
    int3 = gauss_integrate(integrand3, 0.0, 1.0; N=100)

    delta_correction = (Δ^2  / (768 * pi^2)) * int1 +
                       (q^2  / (192 * pi^2)) * int2 -
                       (q^4  / (32 * Δ^2 * pi^2)) * int3

    return (V0 * C0 + delta_correction) * 10^6
end
#  4. 交叉箱图（Cross Box diagram）

function BoxCrossC_unequal(qMeV, m1MeV, m2MeV, DeltaMeV)
    q  = qMeV    / 1000.0
    m1 = m1MeV   / 1000.0
    m2 = m2MeV   / 1000.0
    Δ  = DeltaMeV/ 1000.0

    C0 = 1.0 / (192 * pi^2)

    function V0_box_kernel(m1_, m2_)
        if abs(q) < 0.01 && m1_ == m2_
            m12_ = m1_^2
            return 64*m12_ + 136/3*q^2 + 4.3*q^4/m12_ - 71/140*q^6/m12_^2
        elseif abs(q) < 0.01
            m12_ = m1_^2;  m22_ = m2_^2
            mm2p = m12_ + m22_;  mm2m = m12_ - m22_
            mm   = m12_ * m22_
            m14_ = m12_^2;  m24_ = m22_^2
            m16_ = m12_^3;  m26_ = m22_^3
            m18_ = m14_^2;  m28_ = m24_^2
            mm4p = m14_ + m24_;  mm4m = m14_ - m24_
            mm6m = m16_ - m26_;  mm8m = m18_ - m28_
            mm8p = m18_ + m28_
            logm = (m1_ != m2_) ? log(m1_/m2_) : 0.0
            V00  = 9.5*mm2p + (33*mm4p + 24*mm)*logm/mm2m
            V02  = (169*mm6m - 87*mm*mm2m +
                    6*mm2p*(23*mm4p - 116*mm)*logm) / (6*mm2m^3)
            V04  = 0.75*(13*mm8m - 152*mm*mm4m +
                         8*(4*mm4p*mm + 55*mm^2)*logm) / mm2m^5
            V06  = -0.05*(mm2m*(59*mm8p - 1706*mm*mm4p - 8586*mm^2) +
                          120*mm*mm2p*(4*mm4p + 91*mm)*logm) / mm2m^7
            return V00 + q^2*V02 + q^4*V04 + q^6*V06
        else
            w_   = wf(q, m1_, m2_)
            L_   = Lf(q, m1_, m2_)
            m12_ = m1_^2;  m22_ = m2_^2
            mm2p = m12_ + m22_;  mm2m = m12_ - m22_
            logm = (m1_ != m2_) ? log(m1_/m2_) : 0.0
            mm   = m12_ * m22_
            mm4p = m12_^2 + m22_^2
            V00  = mm2m^2/q^2 + mm2m*(mm2m^2 - 9*q^2*mm2p)*logm/q^4
            V01  = 2/w_^2*(23*q^4 - mm2m^4/q^4 + 56*mm2p*q^2 +
                           8*mm2p*mm2m^2/q^2 + 42*mm4p + 44*mm)*L_
            return V00 + V01
        end
    end

    δ = 1e-6
    dV0 = ( V0_box_kernel(sqrt(m1^2 + δ), m2) - V0_box_kernel(sqrt(m1^2 - δ), m2) +
            V0_box_kernel(m1, sqrt(m2^2 + δ)) - V0_box_kernel(m1, sqrt(m2^2 - δ)) ) / (2*δ)

    V_cross_0 = -0.5 * dV0

    if abs(Δ) < 1e-8
        return V_cross_0 * C0 * 10^6
    end


    integrand_cross1 = x -> begin
        Mq2 = Msq(x, q, m1, m2)
        z   = Mq2 / (4 * Δ^2)
        102.0 + real(dFdz(F3_func, z))
    end

    integrand_cross2 = x -> begin
        Mq2 = Msq(x, q, m1, m2)
        z   = Mq2 / (4 * Δ^2)
        (-10*x^2 + 10*x - 1) / 6.0 * real(dFdz(F1_func, z))
    end

    integrand_cross3 = x -> begin
        Mq2 = Msq(x, q, m1, m2)
        z   = Mq2 / (4 * Δ^2)
        x^2 * (1 - x)^2 * real(dFdz(F2_func, z))
    end

    int1 = gauss_integrate(integrand_cross1, 0.0, 1.0; N=100)
    int2 = gauss_integrate(integrand_cross2, 0.0, 1.0; N=100)
    int3 = gauss_integrate(integrand_cross3, 0.0, 1.0; N=100)

    delta_correction = -Δ^2 / (96 * pi^2) * int1 +
                        q^2  / ( 4 * pi^2) * int2 -
                        q^4  / ( 4 * Δ^2 * pi^2) * int3

    return (V_cross_0 * C0 + delta_correction) * 10^6
end