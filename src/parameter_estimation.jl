# ML parameter estimation with filter result log-likelihood (via Nelder-Mead)

function fit{T}(y::Matrix{T}, build::Function, theta0::Vector{T};
          u::Array{T}=zeros(size(y,1), build(theta0).nu))
    objective(theta) = kalman_filter(y, build(theta), u=u).loglik
    kfit = Optim.optimize(objective, theta0)
    return (kfit.minimum, build(kfit.minimum), filter(y, build(kfit.minimum)))
end #fit

# Expectation-Maximization (EM) parameter estimation

function fit{T}(y::Array{T}, pmodel::ParametrizedSSM, params::SSMParameters;
          u::Array{T}=zeros(size(y,1), size(model.A, 2)), eps::Float64=1e-6, niter::Int=typemax(Int))

    n = size(y, 1)
    @assert n == size(u, 1)

    y_orig = copy(y)
    y = y'
    y_notnan = !isnan(y)
    y = y .* y_notnan

    u_orig = copy(u)
    u = u'

    function em_kernel!{T}(params::SSMParameters{T})

        estimate_A   = length(params.A) > 0
        estimate_B   = length(params.B) > 0
        estimate_Q   = length(params.Q) > 0
        estimate_C   = length(params.C) > 0
        estimate_D   = length(params.D) > 0
        estimate_R   = length(params.R) > 0
        estimate_x1  = length(params.x1) > 0
        estimate_P1  = length(params.P1) > 0

        m = pmodel(params)

        print("Expectations (smoothing)... ")
        tic()

        if estimate_B | estimate_Q
            exs, P, Plag1, loglik = lag1_smooth(y, u, m)
        else
            smoothed = smooth(y, m, u=u)
            exs, P, loglik = smoothed.x', smoothed.V, smoothed.loglik
        end

        toc()
        println("Negative log-likelihood: ", loglik)

        print("Maximizations... ")
        tic()

        if estimate_B | estimate_U | estimate_Q
            phi(t) = (pmodel.G(t)' * pmodel.G(t)) \ pmodel.G(t)'
            Qinv(t) = phi(t)' * inv(pmodel.Q(params.Q)) * phi
        end #if BU

        if estimate_U | estimate_Z | estimate_A | estimate_R
            xi(t) = (pmodel.H(t)' * pmodel.H(t)) \ pmodel.H(t)'
            Rinv(t) = xi(t)' * inv(pmodel.R(params.R)) * xi(t)
        end #if

        if estimate_x1
            Linv = inv(m.P1) #TODO: Degenerate accomodations?
        end #if

        HRH(t) = pmodel.H(t) * pmodel.R(t) * pmodel.H(t)'
        HRH_nonzero_rows(t) = diag(HRH(t)) .!= 0

        function get_Nt(y_notnan_t::Vector{Bool})
            O = eye(m.ny)[find(y_notnan_t & HRH_nonzero_rows), :]
            return I - HRH * O' * inv(O * HRH * O') * O
        end #get_Nt

        ex  = exs[:, 1]
        exx = zeros(m.nx, m.nx)

        yt = y[:, 1]
        Nt = get_Nt(y_notnan[:, 1])
        ey = yt - Nt * (yt - m.C(1) * exs[:, 1] - m.D(1) * u[:, 1])

        Uut = m.B(1) * u[:, 1]
        Aut = m.D(1) * u[:, 1]
        
        if estimate_A # B setup
            beta_S1 = zeros(length(params.B), length(params.B))
            beta_S2 = zeros(params.B)
        end #if B

        if estimate_B # U setup
            deterministic = all(model.G(1) .== 0, 2)
            OQ0, OQp = speye(m.nx)[find(deterministic), :], speye(m.nx)[find(!deterministic), :]

            nu_kp = kron(u[:, 1]', eye(m.nx))
            fU = nu_kp * pmodel.A.f 
            DU = nu_kp * pmodel.A.D 

            Idt = eye(m.nx)
            B_  = eye(m.nx)
            f_  = nu_kp * pmodel.A.f * 0
            D_  = nu_kp * pmodel.D.f * 0
            M   = m.B(1) .!= 0
            M_t = copy(M)
            EX0 = exs[:, 1]
            potential_deterministic_rows = all((OQ0 * M_t * OQp') .== 0, 2)[:]

            Dt1 = ey - m.C(1) * (I - Idt) * ex - m.C(1) * Idt * (B_ * EX0 + f_) - Aut 
            Dt2 = m.C(1) * Idt * D_
            Dt3 = zeros(pmodel.nx) 
            Dt4 = DU * 0

            Idt1 = Idt
            Bt1_ = B_
            ft1_ = f_
            Dt1_ = D_ 

            nu_S1 = Dt2' * Rinv * Dt2
            nu_S2 = Dt2' * Rinv * Dt1
        end #if

        if estimate_Q # Q setup
            q_S1 = zeros(length(params.Q), length(params.Q))
            q_S2 = zeros(params.Q)
        end #if

        if estimate_C # Z setup
            zeta_S1 = zeros(length(params.C), length(params.C)) 
            zeta_S2 = zeros(params.C)
        end #if

        if estimate_D # A setup
            alpha_S1 = zeros(length(params.D), length(params.D)) 
            alpha_S2 = zeros(params.D)
        end #if

        if estimate_R # R setup
            r_S1 = zeros(length(params.R), length(params.R))
            r_S2 = zeros(params.R)
        end #if

        for t in 1:n

            ex_prev = ex
            ex      = exs[:, t]
            At      = m.A(t)
            Vt      = V[:, :, t]
            ut      = u[:, t]

            if estimate_A | estimate_Q | estimate_C

                exx_prev = exx
                exx = Vt + ex * ex'

                if (estimate_A | estimate_Q) && t > 1

                    exx1 = Vlag1[:, :, t] + ex * ex_prev'
                    Uut_prev = Uut
                    Uut = m.B(t) * ut

                    if estimate_A
                        beta_kp = kron(exx_prev, Qinv)
                        beta_S1 += pmodel.B.D' * beta_kp * pmodel.B.D 
                        beta_S2 += pmodel.B.D' * (vec(Qinv * exx1) -
                                      beta_kp * pmodel.B.f - vec(Qinv * Uut_prev * ex_prev'))
                    end #if B

                    if estimate_Q
                        q_S1 += pmodel.Q.D' * pmodel.Q.D
                        q_S2 += pmodel.Q.D' * vec(phi(t) * (exx - exx1 * At' -
                                    At * exx1' - ex * Uut_prev' - Uut_prev * ex' + At * exx_prev * At' +
                                    At * ex_prev * Uut_prev' + Uut_prev * ex_prev' * At' + Uut_prev * Uut_prev') * phi(t)')
                    end #if Q

                end #if BQ

            end #if BQZ

            if estimate_Z | estimate_R | estimate_U | estimate_A
                yt = y[:, t]
                Ct = m.C(t)
                Nt = get_Nt(y_notnan[:, t])
                Aut = m.D(t) * ut
                ey = yt - Nt * (yt - Ct * ex - Aut)

                if estimate_Z | estimate_R

                    eyx = Nt * Ct * Vt + ey * ex'

                    if estimate_Z
                        zeta_kp = kron(exx, Rinv)
                        zeta_S1 += pmodel.Z.D' * zeta_kp * pmodel.Z.D
                        zeta_S2 += pmodel.Z.D' * (vec(Rinv * eyx) - zeta_kp * pmodel.Z.f -
                                      vec(Rinv * Aut * ex'))
                    end #if Z

                    if estimate_R
                        I2 = diagm(!y_notnan[:, t])
                        eyy = I2 * (Nt * HRH(t)' + Nt * Ct * Vt * Ct' * Nt') * I2 + ey * ey'
                        r_S1 += pmodel.R.D' * pmodel.R.D
                        r_S2 += pmodel.R.D' * vec(xi(t) * (eyy - eyx * Ct' -
                                    Ct * eyx' - ey * Aut' - Aut * ey' + Ct * exx * Ct' +
                                    Ct * ex * Aut' + Aut * ex' * Ct' + Aut * Aut') * xi(t)')
                    end #if R

                end #if ZR

                if estimate_U

                    potential_deterministic_rows = all((OQ0 * M_t * OQp') .== 0, 2)[:]
                    Idt = diagm(OQ0' * potential_deterministic_rows)
                    nu_kp = kron(ut', eye(m.nx))
                    B_ = At * Bt1_
                    fU = nu_kp * spU_f
                    DU = nu_kp * spU_D
                    f_ = At * ft1_ + fU
                    D_ = At * Dt1_ + DU 

                    Dt1 = ey - Ct * (I - Idt) * ex - Ct * Idt * (B_ * EX0 + f_) - Aut 
                    Dt2 = Ct * Idt * D_
                    Dt3 = ex - At * (I - Idt1) * ex_prev -
                                At * Idt1 * (Bt1_ * EX0 + ft1_) - fU 
                    Dt4 = DU + At * Idt1 * Dt1_ 

                    nu_S1 += Dt4' * spQinv * Dt4 + Dt2' * Rinv * Dt2
                    nu_S2 += Dt4' * spQinv * Dt3 + Dt2' * Rinv * Dt1 

                    t <= (pmodel.nx +1) ? M_t *= M : nothing
                    Idt1 = Idt
                    Bt1_ = B_
                    ft1_ = f_
                    Dt1_ = D_ 

                end #if U

                if estimate_A
                    alpha_kp = kron(ut', eye(m.ny))
                    alpha_S1 += pmodel.A.D' * alpha_kp' * Rinv * alpha_kp * pmodel.A.D
                    alpha_S2 += pmodel.A.D' * alpha_kp' * Rinv * 
                                    (ey - Ct * ex - alpha_kp * pmodel.A.f)
                end #if A

            end #if ZRUA

        end #for

        params.A[:]  = estimate_A ? beta_S1 \ beta_S2    : T[]
        params.B[:]  = estimate_B ? nu_S1 \ nu_S2        : T[]
        params.Q[:]  = estimate_Q ? q_S1 \ q_S2          : T[]

        params.C[:]  = estimate_C ? zeta_S1 \ zeta_S2    : T[]
        params.D[:]  = estimate_D ? alpha_S1 \ alpha_S2  : T[]
        params.R[:]  = estimate_R ? r_S1 \ r_S2          : T[]

        params.x1[:] = estimate_x1 ? (pmodel.x1.D' * Linv * pmodel.x1.D) \
                        pmodel.x1.D' * Linv * (exs[:, 1] - pmodel.x1.f) : T[]
        params.P1[:] = estimate_P1 ? (pmodel.V1.D' * pmodel.V1.D) \ pmodel.V1.D' *
                        vec(V[1] + exs[:, 1] * exs[:, 1]' -
                        exs[:, 1]*m.x1' - m.x1*exs[:, 1]' + m.x1*m.x1') : T[]

        toc()

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

    # Q, R, V1 default to parametrized as diagonal with independent elements - any other values
    #   are ignored / set to zero 
    Q, Q_params = parametrize_diag(diag(model.V(1)))
    R, R_params = parametrize_diag(diag(model.W(1)))
    P1, P1_params = parametrize_diag(diag(model.P1))

    pmodel = ParametrizedSSM(A, Q, C, R, x1, P1, B=B, D=D)
    params = SSMParameters(A_params, B_params, Q_params, C_params, D_params, R_params, x1_params, V1_params)
    fit(y, pmodel, params, u=u, eps=eps, niter=niter)

end #fit

