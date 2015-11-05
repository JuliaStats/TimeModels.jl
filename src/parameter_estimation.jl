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
    I_nu = eye(pmodel.nu)

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

    estimate_A   = length(params.A) > 0
    estimate_B   = length(params.B) > 0
    estimate_Q   = length(params.Q) > 0
    estimate_C   = length(params.C) > 0
    estimate_D   = length(params.D) > 0
    estimate_R   = length(params.R) > 0
    estimate_x1  = length(params.x1) > 0
    estimate_P1  = length(params.P1) > 0


    function em_kernel!{T}(params::SSMParameters{T})

        m = pmodel(params)

        #print("Expectations (smoothing)... ")
        #tic()

        if estimate_A | estimate_Q
            exs, P, Plag1, loglik = lag1_smooth(y_orig, u_orig, m)
        else
            smoothed = kalman_smooth(y_orig, m, u=u_orig)
            exs, P, loglik = smoothed.smoothed', smoothed.error_cov, smoothed.loglik
        end

        #toc()
        #println("Negative log-likelihood: ", loglik)

        #print("Maximizations... ")
        #tic()

        if estimate_A | estimate_B | estimate_Q
            phi(t)  = (pmodel.G(t)' * pmodel.G(t)) \ pmodel.G(t)' #Can be moved out
            Qinv    = inv(pmodel.Q(params.Q))  
            Q_      = all_x_deterministic ? I0_nx : phi(1)' * Qinv * phi(1)
        end #if ABQ

        if estimate_B | estimate_C | estimate_D | estimate_R
            xi(t) = (pmodel.H(t)' * pmodel.H(t)) \ pmodel.H(t)' #Can be moved out
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

        But = m.B(1) * u[:, 1]
        Dut = m.D(1) * u[:, 1]
        
        if estimate_A # A setup
            A_S1 = zeros(length(params.A), length(params.A))
            A_S2 = zeros(params.A)
        end #if A

        if estimate_B # B setup
            pkp = kron(I_nu, pmodel.B1(1))
            f_Bt = pkp * pmodel.B2.f
            D_Bt = pkp * pmodel.B2.D

            B_kp = kron(u[:, 1]', I_nx)
            fB = B_kp * f_Bt
            DB = B_kp * D_Bt

            Idt = I_nx
            A_  = I_nx
            f_  = B_kp * f_Bt * 0
            D_  = B_kp * D_Bt * 0
            M   = 1 * (m.A(1) .!= 0) #Assumption: constant position of nonzero values in A
            M_t = copy(M)
            EX0 = exs[:, 1]

            Dt1 = ey - m.C(1) * (I - Idt) * ex - m.C(1) * Idt * (A_ * EX0 + f_) - Dut
            Dt2 = m.C(1) * Idt * D_
            Dt3 = zeros(pmodel.nx)
            Dt4 = DB * 0

            Idt1 = Idt
            At1_ = A_
            ft1_ = f_
            Dt1_ = D_ 

            B_S1 = Dt2' * R_ * Dt2
            B_S2 = Dt2' * R_ * Dt1
        end #if B

        if estimate_Q # Q setup
            Q_S1 = zeros(length(params.Q), length(params.Q))
            Q_S2 = zeros(params.Q)
        end #if

        if estimate_C # C setup
            C_S1 = zeros(length(params.C), length(params.C))
            C_S2 = zeros(params.C)
        end #if

        if estimate_D # D setup
            D_S1 = zeros(length(params.D), length(params.D))
            D_S2 = zeros(params.D)
        end #if

        if estimate_R # R setup
            R_S1 = zeros(length(params.R), length(params.R))
            R_S2 = zeros(params.R)
        end #if

        for t in 1:n

            At      = m.A(t)
            Pt      = P[:, :, t]
            ut      = u[:, t]

            ex_prev = ex
            ex      = exs[:, t]

            exx_prev  = exx
            exx       = Pt + ex * ex'

            !all_x_deterministic && (estimate_A || estimate_B || estimate_Q) && (Q_ = phi(t)' * Qinv * phi(t))
            !all_y_deterministic && (estimate_B || estimate_C || estimate_D || estimate_R) && (R_ = xi(t)' * Rinv * xi(t))

            if estimate_A || estimate_Q || estimate_C

                if (estimate_A || estimate_Q) && t > 1

                    exx1 = Plag1[:, :, t] + ex * ex_prev'
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

                    Idt = Id(M_t)
                    B_kp = kron(ut', I_nx)
                    A_ = At * At1_
                    fB = B_kp * f_Bt
                    DB = B_kp * D_Bt
                    f_ = At * ft1_ + fB
                    D_ = At * Dt1_ + DB

                    Dt1 = ey - Ct * (I - Idt) * ex - Ct * Idt * (A_ * EX0 + f_) - Dut
                    Dt2 = Ct * Idt * D_
                    Dt3 = ex - At * (I - Idt1) * ex_prev -
                                At * Idt1 * (At1_ * EX0 + ft1_) - fB
                    Dt4 = DB + At * Idt1 * Dt1_

                    B_S1 += Dt4' * Q_ * Dt4 + Dt2' * R_ * Dt2
                    B_S2 += Dt4' * Q_ * Dt3 + Dt2' * R_ * Dt1

                    t <= (pmodel.nx +1) ? M_t *= M : nothing
                    Idt1 = Idt
                    At1_ = A_
                    ft1_ = f_
                    Dt1_ = D_ 

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

    while (niter > 0) & (fraction_change(ll_prev, ll) > eps)
        ll_prev = ll 
        ll      = em_kernel!(params)
        niter -= 1
    end #while

    niter > 0 ? nothing :
        warn("Parameter estimation timed out - results may have failed to converge")

    return params, pmodel(params)
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

