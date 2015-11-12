# ML parameter estimation with filter result log-likelihood (via Nelder-Mead)

function fit{T}(y::Matrix{T}, build::Function, theta0::Vector{T};
          u::Array{T}=zeros(size(y,1), build(theta0).nu))
    objective(theta) = kalman_filter(y, build(theta), u=u).loglik
    kfit = Optim.optimize(objective, theta0)
    return (kfit.minimum, build(kfit.minimum), filter(y, build(kfit.minimum)))
end #fit

# Expectation-Maximization (EM) parameter estimation

function fit{T}(y::Array{T}, pmodel::ParametrizedSSM, params::SSMParameters;
          u::Array{T}=zeros(size(y,1), pmodel.nu), eps::Float64=1e-6, niter::Int=typemax(Int))

    n = size(y, 1)
    @assert n == size(u, 1)

    y_orig = copy(y)
    y = y'
    y_notnan = !isnan(y)
    y = y .* y_notnan

    u_orig = copy(u)
    u = u'

    I_nx = eye(pmodel.nx)
    I0_nx = 0I_nx

    I_ny = eye(pmodel.ny)
    I0_ny = 0I_ny

    I_nu = speye(pmodel.nu)

    # Requirement: zeros-rows in G and H remain consistent
    x_deterministic = all(pmodel.G(1) .== 0, 2)
    all_x_deterministic = all(x_deterministic)
    y_deterministic = all(pmodel.H(1) .== 0, 2)
    all_y_deterministic = all(y_deterministic)

    if any(x_deterministic)
        OQ0, OQp = I_nx[find(x_deterministic), :], I_nx[find(!x_deterministic), :]
        Id(M_t::Array{Int,2}) = diagm(OQ0' * all((OQ0 * M_t * OQp') .== 0, 2)[:])
    else
        Id(_::Array{Int,2}) = I0_nx
    end

    phi(t)  = (pmodel.G(t)' * pmodel.G(t)) \ pmodel.G(t)'
    xi(t) = (pmodel.H(t)' * pmodel.H(t)) \ pmodel.H(t)'

    na, nb, nq, nc, nd, nr, nx1, nP1 =
        map(length, Vector[params.A, params.B, params.Q,
                        params.C, params.D, params.R, params.x1, params.P1])

    estimate_A, estimate_B, estimate_Q, estimate_C, estimate_D, estimate_R,
        estimate_x1, estimate_P1 = map(x->x>0, [na, nb, nq, nc, nd, nr, nx1, nP1])

    function em_kernel!{T}(params::SSMParameters{T})

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

        if estimate_A | estimate_B | estimate_Q
            Qinv    = inv(pmodel.Q(params.Q))  
            Q_      = all_x_deterministic ? I0_nx : phi(1)' * Qinv * phi(1)
        end #if ABQ

        if estimate_B | estimate_C | estimate_D | estimate_R
            Rinv  = inv(pmodel.R(params.R))
            R_    = all_y_deterministic ? I0_ny : xi(1)' * Rinv * xi(1)
        end #if

        if estimate_x1
            Linv = inv(m.P1) #TODO: Degenerate accomodations?
        end #if

        HRH(t) = pmodel.H(t) * pmodel.R(params.R) * pmodel.H(t)'
        HRH_nonzero_rows(t) = diag(HRH(t)) .!= 0

        function N(t::Int)
            HRHt = HRH(t)
            O = I_ny[find(y_notnan[:, t] & HRH_nonzero_rows(t)), :]
            return I - HRHt * O' * inv(O * HRHt * O') * O
        end #N

        ex  = exs[:, 1]
        exx = zeros(m.nx, m.nx)

        yt = y[:, 1]
        Nt = N(1)
        ey = yt - Nt * (yt - m.C(1) * exs[:, 1] - m.D(1) * u[:, 1])

        At  = m.A(1)
        But = m.B(1) * u[:, 1]
        Dut = m.D(1) * u[:, 1]
        
        estimate_A && ((A_S1, A_S2) = (zeros(na, na), zeros(na)))

        if estimate_B # B setup

            x1 = m.x1
            pmodel_B2_D = sparse(pmodel.B2.D)

            Δt1, Δt2 = zeros(m.nx, nb), zeros(m.nx)
            Δt3, Δt4 = zeros(m.ny, nb), zeros(m.ny)

            M   = 1 * (m.A(1) .!= 0) #Assumption: constant position of nonzero values in A
            M_t = copy(M)

            Idt     = Idt1      = I_nx
            At1_    = At2_      = I_nx
            f_Bu_t1 = f_Bu_t2   = zeros(m.nx)
            D_Bu_t1 = D_Bu_t2   = zeros(m.nx, nb)
            ku_f_Bt = ku_f_Bt1  = zeros(m.nx)
            ku_D_Bt = ku_D_Bt1  = zeros(m.nx, nb)

            B_S1, B_S2 = zeros(nb, nb), zeros(nb)

        end #if B

        estimate_Q && ((Q_S1, Q_S2) = (zeros(nq, nq), zeros(nq)))
        estimate_C && ((C_S1, C_S2) = (zeros(nc, nc), zeros(nc)))
        estimate_D && ((D_S1, D_S2) = (zeros(nd, nd), zeros(nd)))
        estimate_R && ((R_S1, R_S2) = (zeros(nr, nr), zeros(nr)))

        for t in 1:n

            At_prev = At #TODO: Use this in the appropriate places?
            At      = m.A(t)
            ut      = u[:, t]
            Pt      = P[t]

            ex_prev = ex
            ex      = exs[:, t]

            exx_prev  = exx
            exx       = Pt + ex * ex'

            !all_x_deterministic && (estimate_A || estimate_B || estimate_Q) && (Q_ = phi(t)' * Qinv * phi(t))
            !all_y_deterministic && (estimate_B || estimate_C || estimate_D || estimate_R) && (R_ = xi(t)' * Rinv * xi(t))

            if estimate_A || estimate_Q || estimate_C

                if (estimate_A || estimate_Q) && t > 1

                    exx1 = Plag1[t] + ex * ex_prev'
                    But_prev = But
                    But = m.B(t) * ut

                    if estimate_A
                        pkp = kron(pmodel.A3(t)', pmodel.A1(t))
                        D_At = pkp * pmodel.A2.D
                        f_At = pkp * pmodel.A2.f
                        A_kp = kron(exx_prev, Q_)
                        A_S1 += D_At' * A_kp * D_At
                        A_S2 += D_At' * (vec(Q_ * exx1) -
                                      A_kp * f_At - vec(Q_ * But_prev * ex_prev'))
                    end #if B

                    if estimate_Q
                        Q_S1 += pmodel.Q.D' * pmodel.Q.D
                        Q_S2 += pmodel.Q.D' * vec(phi(t) * (exx - exx1 * At' -
                                    At * exx1' - ex * But_prev' - But_prev * ex' + At * exx_prev * At' +
                                    At * ex_prev * But_prev' + But_prev * ex_prev' * At' + But_prev * But_prev') * phi(t)')
                    end #if Q

                end #if BQ

            end #if BQZ

            if estimate_C | estimate_R | estimate_B | estimate_D
                yt = y[:, t]
                Ct = m.C(t)
                Nt = N(t)
                Dut = m.D(t) * ut
                ey = yt - Nt * (yt - Ct * ex - Dut)

                if estimate_C | estimate_R

                    eyx = Nt * Ct * Pt + ey * ex'

                    if estimate_C
                        pkp = kron(pmodel.C3(t)', pmodel.C1(t))
                        f_Ct = pkp * pmodel.C2.f
                        D_Ct = pkp * pmodel.C2.D
                        C_kp = kron(exx, R_)
                        C_S1 += D_Ct' * C_kp * D_Ct
                        C_S2 += D_Ct' * (vec(R_ * eyx) - C_kp * f_Ct - vec(R_ * Dut * ex'))
                    end #if Z

                    if estimate_R
                        I2 = diagm(!y_notnan[:, t])
                        eyy = I2 * (Nt * HRH(t) + Nt * Ct * Pt * Ct' * Nt') * I2 + ey * ey'
                        R_S1 += pmodel.R.D' * pmodel.R.D
                        R_S2 += pmodel.R.D' * vec(xi(t) * (eyy - eyx * Ct' -
                                    Ct * eyx' - ey * Dut' - Dut * ey' + Ct * exx * Ct' +
                                    Ct * ex * Dut' + Dut * ex' * Ct' + Dut * Dut') * xi(t)')
                    end #if R

                end #if ZR

                if estimate_B

                    if t > 1
                        Δt1 = At_prev * Idt1 * D_Bu_t2 + ku_D_Bt1
                        Δt2 = ex - At_prev * (I - Idt1) * ex_prev - At_prev * Idt1 * (At2_ * x1 + f_Bu_t2) - ku_f_Bt1
                    end

                    Δt3 = Ct * Idt * D_Bu_t1
                    Δt4 = ey - Ct * (I - Idt) * ex - Ct * Idt * (At1_ * x1 + f_Bu_t1) - Dut

                    B_S1 += Δt1' * Q_ * Δt1 + Δt3' * R_ * Δt3
                    B_S2 += Δt1' * Q_ * Δt2 + Δt3' * R_ * Δt4

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

                    ku_D_Bt   = kuB * pmodel_B2_D
                    ku_D_Bt1  = ku_D_Bt

                    D_Bu_t2   = D_Bu_t1
                    D_Bu_t1   = ku_D_Bt + At * D_Bu_t1

                end #if B

                if estimate_D
                    pkp = kron(I_nu, pmodel.D1(t))
                    f_Dt = pkp * pmodel.D2.f
                    D_Dt = pkp * pmodel.D2.D
                    D_kp = kron(ut', I_ny)
                    D_S1 += D_Dt' * D_kp' * R_ * D_kp * D_Dt
                    D_S2 += D_Dt' * D_kp' * R_ * (ey - Ct * ex - D_kp * f_Dt)
                end #if D

            end #if CRBD

        end #for

        params.A[:]  = estimate_A ? A_S1 \ A_S2 : T[]
        params.B[:]  = estimate_B ? B_S1 \ B_S2 : T[]
        params.Q[:]  = estimate_Q ? Q_S1 \ Q_S2 : T[]

        params.C[:]  = estimate_C ? C_S1 \ C_S2 : T[]
        params.D[:]  = estimate_D ? D_S1 \ D_S2 : T[]
        params.R[:]  = estimate_R ? R_S1 \ R_S2 : T[]

        params.x1[:] = estimate_x1 ? (pmodel.x1.D' * Linv * pmodel.x1.D) \
                        pmodel.x1.D' * Linv * (exs[:, 1] - pmodel.x1.f) : T[]
        params.P1[:] = estimate_P1 ? (pmodel.P1.D' * pmodel.P1.D) \ pmodel.P1.D' *
                        vec(V[1] + exs[:, 1] * exs[:, 1]' -
                        exs[:, 1]*m.x1' - m.x1*exs[:, 1]' + m.x1*m.x1') : T[]

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

function fit{T}(y::Array{T}, model::StateSpaceModel{T}; u::Array{T}=zeros(size(y,1), model.nu),
              eps::Float64=1e-6, niter::Int=typemax(Int))

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

