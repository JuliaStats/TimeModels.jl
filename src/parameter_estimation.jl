# ML parameter estimation with filter result log-likelihood (via Nelder-Mead)

function fit(y::Matrix{T}, build::Function, theta0::Vector{T};
       u::Array{T}=zeros(size(y,1), build(theta0).nu)) where T
    objective(theta) = kalman_filter(y, build(theta), u=u).loglik
    kfit = Optim.optimize(objective, theta0)
    return (kfit.minimum, build(kfit.minimum), filter(y, build(kfit.minimum)))
end #fit

# Expectation-Maximization (EM) parameter estimation

function fit(y::Array{T}, pmodel::ParametrizedSSM, params::SSMParameters;
       u::Array{T}=zeros(size(y,1), pmodel.nu), eps::Float64=1e-6, niter::Int=typemax(Int)) where T

    n = size(y, 1)
    @assert n == size(u, 1)

    y_orig = copy(y)
    y = y'
    y_notnan = (!).(isnan.(y))
    y = y .* y_notnan

    u_orig = copy(u)
    u = u'

    I_nx = speye(pmodel.nx)
    I0_nx = zeros(I_nx)

    I_ny = speye(pmodel.ny)
    I0_ny = zeros(I_ny)

    I_nu = speye(pmodel.nu)

    # Requirement: zeros-rows in G and H remain consistent
    x_deterministic       = all(pmodel.G(1) .== 0, 2)
    all_x_deterministic   = all(x_deterministic)

    y_deterministic       = all(pmodel.H(1) .== 0, 2)
    all_y_deterministic   = all(y_deterministic)

    x1_deterministic      = all(pmodel.J .== 0, 2) |> full |> vec
    all_x1_deterministic  = all(x1_deterministic)
    Id_x1                 = spdiagm(x1_deterministic)

    if any(x_deterministic)
        OQ0, OQp = I_nx[find(x_deterministic), :], I_nx[find((!).(x_deterministic)), :]
        # Id(M_t::Array{Int,2}) = spdiagm(OQ0' * all((OQ0 * M_t * OQp') .== 0, 2)[:])
        Id = M_t -> spdiagm(OQ0' * all((OQ0 * M_t * OQp') .== 0, 2)[:])
    else
        # Id(_::Array{Int,2}) = I0_nx
        Id = t -> I0_nx
    end

    phi(t)  = (pmodel.G(t)' * pmodel.G(t)) \ pmodel.G(t)'
    xi(t)   = (pmodel.H(t)' * pmodel.H(t)) \ pmodel.H(t)'
    Pi      = (pmodel.J' * pmodel.J) \ pmodel.J'

    na, nb, nq, nc, nd, nr, nx1, ns =
        map(length, Vector[params.A, params.B, params.Q,
                        params.C, params.D, params.R, params.x1, params.S])

    estimate_A, estimate_B, estimate_Q, estimate_C, estimate_D, estimate_R,
        estimate_x1, estimate_S = map(x->x>0, [na, nb, nq, nc, nd, nr, nx1, ns])

    function em_kernel!(params::SSMParameters{T}) where T

        m = pmodel(params)

        #=
        print("Expectations (smoothing)... ")
        tic()
        =#

        if estimate_A | estimate_Q
            exs, P, Plag1, loglik = lag1_smooth(y_orig, u_orig, m)
        else
            smoothed = kalman_smooth(y_orig, m, u=u_orig)
            exs, P, loglik = smoothed.smoothed', smoothed.error_cov, smoothed.loglik
        end

        #=
        toc()
        println("Negative log-likelihood: ", loglik)

        print("Maximizations... ")
        tic()
        =#

        if estimate_A || estimate_B || estimate_Q || estimate_x1
            Qinv    = inv(pmodel.Q(params.Q))
            Vinv_   = all_x_deterministic ? I0_nx : phi(1)' * Qinv * phi(1)
        end #if ABQ

        if estimate_B || estimate_C || estimate_D || estimate_R || estimate_x1
            Rinv  = inv(pmodel.R(params.R))
            Winv_ = all_y_deterministic ? I0_ny : xi(1)' * Rinv * xi(1)
        end #if BCDRx1

        if estimate_x1
            Sinv  = inv(pmodel.S(params.S))
            P1inv_   = all_x1_deterministic ? I0_nx : Pi' * Sinv * Pi
        end #if x1

        HRH(t) = pmodel.H(t) * pmodel.R(params.R) * pmodel.H(t)'
        HRH_nonzero_rows(t) = diag(HRH(t)) .!= 0

        function N(t::Int)
            HRHt = HRH(t)
            O = I_ny[find(y_notnan[:, t] .& HRH_nonzero_rows(t)), :]
            return I - HRHt * O' * inv(O * HRHt * O') * O
        end #N

        ex  = exs[:, 1]
        exx = I0_nx

        At  = m.A(1)
        ut  = u[:, 1]

        if estimate_B || estimate_x1

            ex1_f = (I_nx - Id_x1) * ex + Id_x1 * pmodel.x1.f
            ex1_D = Id_x1 * pmodel.x1.D
            ex1 = ex1_f + ex1_D * params.x1

            M   = 1 * (m.A(1) .!= 0) #TODO: Sparse? Assumption: constant position of nonzero values in A
            M_t = copy(M)
            Idt     = Idt1      = I_nx
            At1_    = At2_      = I_nx
            f_Bu_t1 = f_Bu_t2   = zeros(m.nx) # Sparse initialization?
            D_Bu_t1 = D_Bu_t2   = zeros(m.nx, nb)
            ku_f_Bt = ku_f_Bt1  = zeros(m.nx)
            ku_D_Bt = ku_D_Bt1  = zeros(m.nx, nb)

        end # if Bx1

        estimate_A && ((A_S1, A_S2) = (zeros(na, na), zeros(na)))

        if estimate_B # B setup

            Δt1, Δt2 = zeros(m.nx, nb), zeros(m.nx)
            Δt3, Δt4 = zeros(m.ny, nb), zeros(m.ny)

            B_S1, B_S2 = zeros(nb, nb), zeros(nb)

        end #if B

        estimate_Q && ((Q_S1, Q_S2) = (zeros(nq, nq), zeros(nq)))
        estimate_C && ((C_S1, C_S2) = (zeros(nc, nc), zeros(nc)))
        estimate_D && ((D_S1, D_S2) = (zeros(nd, nd), zeros(nd)))
        estimate_R && ((R_S1, R_S2) = (zeros(nr, nr), zeros(nr)))

        if estimate_x1 #x1 setup

            Δ5t, Δ6t = zeros(m.nx, nx1), zeros(m.nx)
            Δ7t, Δ8t = zeros(m.ny, nx1), zeros(m.ny)

            x1_S1 = pmodel.x1.D' * P1inv_ * pmodel.x1.D
            x1_S2 = pmodel.x1.D' * P1inv_ * (ex - pmodel.x1.f)

        end # if x1

        for t in 1:n

            (t > 1) && (estimate_A || estimate_Q || estimate_x1) && (But_prev = m.B(t-1) * ut)

            At_prev = At #TODO: Use this in the appropriate places?
            At      = m.A(t)

            ut      = u[:, t]
            Pt      = P[t]

            ex_prev = ex
            ex      = exs[:, t]

            exx_prev  = exx
            exx       = Pt + ex * ex'

            !all_x_deterministic && (estimate_A || estimate_B || estimate_Q || estimate_x1) && (Vinv_ = phi(t)' * Qinv * phi(t))

            if (estimate_A || estimate_Q) && t > 1

                exx1 = Plag1[t] + ex * ex_prev'

                if estimate_A
                    pkp = kron(pmodel.A3(t)', pmodel.A1(t))
                    D_At = pkp * pmodel.A2.D
                    f_At = pkp * pmodel.A2.f
                    A_kp = kron(exx_prev, Vinv_)
                    A_S1 += D_At' * A_kp * D_At
                    A_S2 += D_At' * (vec(Vinv_ * exx1) -
                                  A_kp * f_At - vec(Vinv_ * But_prev * ex_prev'))
                end #if A

                if estimate_Q
                    Q_S1 += pmodel.Q.D' * pmodel.Q.D
                    Q_S2 += pmodel.Q.D' * vec(phi(t) * (exx - exx1 * At' -
                                At * exx1' - ex * But_prev' - But_prev * ex' + At * exx_prev * At' +
                                At * ex_prev * But_prev' + But_prev * ex_prev' * At' + But_prev * But_prev') * phi(t)')
                end #if Q

            end #if AQ

            if estimate_B || estimate_C || estimate_D || estimate_R || estimate_x1

                yt = y[:, t]
                Ct = m.C(t)
                Nt = N(t)
                Dut = m.D(t) * ut
                ey = yt - Nt * (yt - Ct * ex - Dut)
                !all_y_deterministic && (Winv_ = xi(t)' * Rinv * xi(t))

                if estimate_C || estimate_R

                    eyx = Nt * Ct * Pt + ey * ex'

                    if estimate_C
                        pkp = kron(pmodel.C3(t)', pmodel.C1(t))
                        f_Ct = pkp * pmodel.C2.f
                        D_Ct = pkp * pmodel.C2.D
                        C_kp = kron(exx, Winv_)
                        C_S1 += D_Ct' * C_kp * D_Ct
                        C_S2 += D_Ct' * (vec(Winv_ * eyx) - C_kp * f_Ct - vec(Winv_ * Dut * ex'))
                    end #if C

                    if estimate_R
                        I2 = diagm(.!y_notnan[:, t])
                        eyy = I2 * (Nt * HRH(t) + Nt * Ct * Pt * Ct' * Nt') * I2 + ey * ey'
                        R_S1 += pmodel.R.D' * pmodel.R.D
                        R_S2 += pmodel.R.D' * vec(xi(t) * (eyy - eyx * Ct' -
                                    Ct * eyx' - ey * Dut' - Dut * ey' + Ct * exx * Ct' +
                                    Ct * ex * Dut' + Dut * ex' * Ct' + Dut * Dut') * xi(t)')
                    end #if R

                end #if CR

                if estimate_B || estimate_x1

                    if estimate_B

                        if t > 1
                            Δt1 = At_prev * Idt1 * D_Bu_t2 + ku_D_Bt1
                            Δt2 = ex - At_prev * (I - Idt1) * ex_prev - At_prev * Idt1 * (At2_ * ex1 + f_Bu_t2) - ku_f_Bt1
                        end

                        Δt3 = Ct * Idt * D_Bu_t1
                        Δt4 = ey - Ct * (I - Idt) * ex - Ct * Idt * (At1_ * ex1 + f_Bu_t1) - Dut

                        B_S1 += Δt1' * Vinv_ * Δt1 + Δt3' * Winv_ * Δt3
                        B_S2 += Δt1' * Vinv_ * Δt2 + Δt3' * Winv_ * Δt4

                    end #if B

                    if estimate_x1

                        if t > 1
                            Δ5t = At_prev * Idt1 * At2_ * ex1_D
                            Δ6t = ex - At_prev * (I - Idt1) * ex_prev - At_prev * Idt1 * (At2_ * ex1_f + f_Bu_t2 + D_Bu_t2 * params.B) - But_prev
                        end

                        Δ7t = Ct * Idt * At1_ * ex1_D
                        Δ8t = ey - Ct * (I - Idt) * ex - Ct * Idt * (At1_ * ex1_f + f_Bu_t1 + D_Bu_t1 * params.B) - Dut

                        x1_S1 += Δ5t' * Vinv_ * Δ5t + Δ7t' * Winv_ * Δ7t
                        x1_S2 += Δ5t' * Vinv_ * Δ6t + Δ7t' * Winv_ * Δ8t

                    end #x1

                    Idt = Id(M_t)
                    Idt1 = Idt
                    t <= pmodel.nx && (M_t *= M)

                    At2_ = At1_
                    At1_ = At * At1_

                    kuB = kron(ut', pmodel.B1(t))

                    ku_f_Bt   = kuB * pmodel.B2.f
                    ku_f_Bt1  = ku_f_Bt

                    f_Bu_t2   = f_Bu_t1
                    f_Bu_t1   = ku_f_Bt + At * f_Bu_t1

                    ku_D_Bt   = kuB * pmodel.B2.D
                    ku_D_Bt1  = ku_D_Bt

                    D_Bu_t2   = D_Bu_t1
                    D_Bu_t1   = ku_D_Bt + At * D_Bu_t1

                end #Bx1

                if estimate_D
                    pkp = kron(I_nu, pmodel.D1(t))
                    f_Dt = pkp * pmodel.D2.f
                    D_Dt = pkp * pmodel.D2.D
                    D_kp = kron(ut', I_ny)
                    D_S1 += D_Dt' * D_kp' * Winv_ * D_kp * D_Dt
                    D_S2 += D_Dt' * D_kp' * Winv_ * (ey - Ct * ex - D_kp * f_Dt)
                end #if D

            end #if BCDRx1

        end #for

        params.A[:]   = estimate_A ? A_S1 \ A_S2 : T[]
        params.B[:]   = estimate_B ? B_S1 \ B_S2 : T[]
        params.Q[:]   = estimate_Q ? Q_S1 \ Q_S2 : T[]

        params.C[:]   = estimate_C ? C_S1 \ C_S2 : T[]
        params.D[:]   = estimate_D ? D_S1 \ D_S2 : T[]
        params.R[:]   = estimate_R ? R_S1 \ R_S2 : T[]

        params.x1[:]  = estimate_x1 ? x1_S1 \ x1_S2 : T[]
        params.S[:]   = estimate_S ? (pmodel.S.D' * pmodel.S.D) \ pmodel.S.D' *
                        vec(Pi * (P[1] + exs[:, 1] * exs[:, 1]' -
                        exs[:, 1]*m.x1' - m.x1*exs[:, 1]' + m.x1*m.x1') * Pi') : T[]

        #toc()

        return loglik

    end #em_kernel

    fraction_change(ll_prev, ll_curr) = isinf(ll_prev) ?
        1 : 2 * (ll_prev - ll_curr) / (ll_prev + ll_curr + 1e-6)

    ll, ll_prev = Inf, Inf

    while (niter > 0) && (fraction_change(ll_prev, ll) > eps)
        ll_prev = ll
        ll      = em_kernel!(params)
        niter -= 1
    end #while

    (niter == 0) && warn("Parameter estimation timed out - results may have failed to converge")

    return params, kalman_smooth(y_orig, pmodel(params), u=u_orig)

end #fit

function fit(y::Array{T}, model::StateSpaceModel{T}; u::Array{T}=zeros(size(y,1), model.nu),
           eps::Float64=1e-6, niter::Int=typemax(Int)) where T

    # B, Z, x1 default to parametrized as fully unconstrained
    A, A_params = parametrize_full(model.A(1))
    C, C_params = parametrize_full(model.C(1))
    x1, x1_params = parametrize_full(model.x1)

    # U, A default to fixed
    B, B_params = parametrize_none(model.B(1))
    D, D_params = parametrize_none(model.D(1))

    # Q, R, P1 default to parametrized as diagonal with independent elements - any other values
    #   are ignored / set to zero 
    Q, Q_params = parametrize_diag(diag(model.V(1)))
    R, R_params = parametrize_diag(diag(model.W(1)))
    P1, P1_params = parametrize_diag(diag(model.P1))

    pmodel = ParametrizedSSM(A, Q, C, R, x1, P1, B=B, D=D)
    params = SSMParameters(A=A_params, B=B_params, Q=Q_params,
                            C=C_params, D=D_params, R=R_params,
                            x1=x1_params, P1_params)
    fit(y, pmodel, params, u=u, eps=eps, niter=niter)

end #fit

