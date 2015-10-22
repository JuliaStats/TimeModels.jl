# ML parameter estimation with filter result log-likelihood (via Nelder-Mead)

function fit{T}(y::Matrix{T}, build::Function, theta0::Vector{T};
          u::Array{T}=zeros(size(y,1), build(theta0).nu))
    objective(theta) = kalman_filter(y, build(theta), u=u).loglik
    kfit = Optim.optimize(objective, theta0)
    return (kfit.minimum, build(kfit.minimum), filter(y, build(kfit.minimum)))
end #fit

# Expectation-Maximization (EM) parameter estimation

function fit{T}(y::Array{T}, model::ParametrizedSSM, params::SSMParameters;
          u::Array{T}=zeros(size(y,1), size(model.A, 2)), eps::Float64=1e-6, niter::Int=typemax(Int))

    function em_kernel{T}(y::Array, u::Array, pmodel::ParametrizedSSM, params::SSMParameters{T})

        function lag1_smooth(y::Array, u::Array, model::StateSpaceModel)
            B_stack = [model.B zeros(model.B); eye(model.nx) zeros(model.B)]
            U_stack = [model.U; zeros(model.U)]
            G_stack = [model.G; zeros(model.G)]
            Z_stack = [model.Z zeros(model.Z)]
            x1_stack = [model.x1; zeros(model.x1)]
            V1_stack = [model.V1 zeros(model.V1); zeros(model.V1) zeros(model.V1)]
            stack_model = StateSpaceModel(B_stack, U_stack, G_stack, model.Q, Z_stack, model.A, model.H, model.R, x1_stack, V1_stack)
            stack_smoothed = smooth(y, stack_model, u=u)
            V     = SparseMatrixCSC[Vt[1:model.nx, 1:model.nx] for Vt in stack_smoothed.V]
            Vlag1 = SparseMatrixCSC[Vt[1:model.nx, (model.nx+1):end] for Vt in stack_smoothed.V]
            return stack_smoothed.x[:, 1:model.nx]', V, Vlag1, stack_smoothed.loglik
        end #function

        estimate_B   = length(params.B) > 0
        estimate_U   = length(params.U) > 0
        estimate_Q   = length(params.Q) > 0
        estimate_Z   = length(params.Z) > 0
        estimate_A   = length(params.A) > 0
        estimate_R   = length(params.R) > 0
        estimate_x1  = length(params.x1) > 0
        estimate_V1  = length(params.V1) > 0

        m = pmodel(params)
        spB = sparse(m.B)
        spU = sparse(m.U)

        print("Expectations (smoothing)... ")
        tic()

        if estimate_B | estimate_Q
            exs, V, Vlag1, loglik = lag1_smooth(y, u, m)
        else
            smoothed = smooth(y, m, u=u)
            exs, V, loglik = smoothed.x', smoothed.V, smoothed.loglik
        end

        toc()
        println("Negative log-likelihood: ", loglik)

        y = y'
        u = u'
        n = size(y,2)

        print("Maximizations... ")
        tic()

        y_notnan = !isnan(y)
        y = y .* y_notnan

        if estimate_B | estimate_U | estimate_Q
            phi = (m.G' * m.G) \ m.G'
            Qinv = phi' * inv(m.Q) * phi
            spQinv = sparse(Qinv)
        end #if BU

        if estimate_U | estimate_Z | estimate_A | estimate_R
            xi = (m.H' * m.H) \ m.H'
            Rinv = xi' * inv(m.R) * xi
        end #if

        if estimate_x1
            Linv = inv(m.V1) #TODO: Degenerate accomodations?
        end #if

        HRH = m.H * m.R * m.H'
        HRH_nonzero_rows = diag(HRH) .!= 0

        function get_Nt(y_notnan_t::Vector{Bool})
            O = eye(m.ny)[find(y_notnan_t & HRH_nonzero_rows), :]
            return I - HRH * O' * inv(O * HRH * O') * O
        end #get_Nt

        ex  = exs[:, 1]
        exx = zeros(m.nx, m.nx)

        yt = y[:, 1]
        Nt = get_Nt(y_notnan[:, 1])
        ey = yt - Nt * (yt - m.Z * exs[:, 1] - m.A * u[:, 1])

        Uut = spU * u[:, 1]
        Aut = m.A * u[:, 1]
        
        if estimate_B # B setup
            spB_D = sparse(pmodel.B.D)
            beta_S1 = zeros(length(params.B), length(params.B))
            beta_S2 = zeros(params.B)
        end #if B

        if estimate_U # U setup
            spU_f = sparse(pmodel.U.f)
            spU_D = sparse(pmodel.U.D)

            deterministic = all(m.G .== 0, 2)
            OQ0, OQp = speye(m.nx)[find(deterministic), :], speye(m.nx)[find(!deterministic), :]

            nu_kp = kron(u[:, 1]', speye(m.nx))
            fU = nu_kp * spU_f 
            DU = nu_kp * spU_D 

            Idt = speye(m.nx)
            B_ = speye(m.nx)
            f_ = nu_kp * spU_f * 0
            D_ = nu_kp * spU_D * 0
            M = sparse((m.B .!= 0) * 1.)
            M_t = copy(M)
            EX0 = exs[:, 1]
            potential_deterministic_rows = all((OQ0 * M_t * OQp') .== 0, 2)[:]

            Dt1 = ey - m.Z * (I - Idt) * ex - m.Z * Idt * (B_ * EX0 + f_) - Aut 
            Dt2 = m.Z * Idt * D_
            Dt3 = zeros(model.nx) 
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

        if estimate_Z # Z setup
            zeta_S1 = zeros(length(params.Z), length(params.Z)) 
            zeta_S2 = zeros(params.Z)
        end #if

        if estimate_A # A setup
            alpha_S1 = zeros(length(params.A), length(params.A)) 
            alpha_S2 = zeros(params.A)
        end #if

        if estimate_R # R setup
            r_S1 = zeros(length(params.R), length(params.R))
            r_S2 = zeros(params.R)
        end #if

        for t in 1:n

            ex_prev = ex
            ex      = exs[:, t]
            Vt      = V[t]
            ut      = u[:, t]

            println(Vt)

            if estimate_B | estimate_Q | estimate_Z

                exx_prev = exx
                exx = Vt + ex * ex'

                if (estimate_B | estimate_Q) && t > 1

                    exx1 = Vlag1[t] + ex * ex_prev'
                    Uut_prev = Uut
                    Uut = spU * ut

                    if estimate_B
                        beta_kp = kron(exx_prev, spQinv)
                        beta_S1 += spB_D' * beta_kp * spB_D
                        beta_S2 += pmodel.B.D' * (vec(Qinv * exx1) -
                                      beta_kp * pmodel.B.f - vec(Qinv * Uut_prev * ex_prev'))
                    end #if B

                    if estimate_Q
                        q_S1 += pmodel.Q.D' * pmodel.Q.D
                        q_S2 += pmodel.Q.D' * vec(phi * (exx - exx1 * spB' -
                                    spB * exx1' - ex * Uut_prev' - Uut_prev * ex' + spB * exx_prev * spB' +
                                    spB * ex_prev * Uut_prev' + Uut_prev * ex_prev' * spB' + Uut_prev * Uut_prev') * phi')
                    end #if Q

                end #if BQ

            end #if BQZ

            if estimate_Z | estimate_R | estimate_U | estimate_A
                yt = y[:, t]
                Nt = get_Nt(y_notnan[:, t])
                Aut = m.A * ut
                ey = yt - Nt * (yt - m.Z * ex - Aut)

                if estimate_Z | estimate_R

                    eyx = Nt * m.Z * Vt + ey * ex'

                    if estimate_Z
                        zeta_kp = kron(exx, Rinv)
                        zeta_S1 += pmodel.Z.D' * zeta_kp * pmodel.Z.D
                        zeta_S2 += pmodel.Z.D' * (vec(Rinv * eyx) - zeta_kp * pmodel.Z.f -
                                      vec(Rinv * Aut * ex'))
                    end #if Z

                    if estimate_R
                        I2 = diagm(!y_notnan[:, t])
                        eyy = I2 * (Nt * HRH' + Nt * m.Z * Vt * m.Z' * Nt') * I2 + ey * ey'
                        r_S1 += pmodel.R.D' * pmodel.R.D
                        r_S2 += pmodel.R.D' * vec(xi * (eyy - eyx * m.Z' -
                                    m.Z * eyx' - ey * Aut' - Aut * ey' + m.Z * exx * m.Z' +
                                    m.Z * ex * Aut' + Aut * ex' * m.Z' + Aut * Aut') * xi')
                    end #if R

                end #if ZR

                if estimate_U

                    potential_deterministic_rows = all((OQ0 * M_t * OQp') .== 0, 2)[:]
                    Idt = sparse(diagm(OQ0' * potential_deterministic_rows))
                    nu_kp = kron(ut', speye(m.nx))
                    B_ = spB * Bt1_
                    fU = nu_kp * spU_f
                    DU = nu_kp * spU_D
                    f_ = spB * ft1_ + fU
                    D_ = spB * Dt1_ + DU 

                    Dt1 = ey - m.Z * (I - Idt) * ex - m.Z * Idt * (B_ * EX0 + f_) - Aut 
                    Dt2 = m.Z * Idt * D_
                    Dt3 = ex - spB * (I - Idt1) * ex_prev -
                                spB * Idt1 * (Bt1_ * EX0 + ft1_) - fU 
                    Dt4 = DU + spB * Idt1 * Dt1_ 

                    nu_S1 += Dt4' * spQinv * Dt4 + Dt2' * Rinv * Dt2
                    nu_S2 += Dt4' * spQinv * Dt3 + Dt2' * Rinv * Dt1 

                    t <= (model.nx +1) ? M_t *= M : nothing
                    Idt1 = Idt
                    Bt1_ = B_
                    ft1_ = f_
                    Dt1_ = D_ 

                end #if U

                if estimate_A
                    alpha_kp = kron(ut', eye(m.ny))
                    alpha_S1 += pmodel.A.D' * alpha_kp' * Rinv * alpha_kp * pmodel.A.D
                    alpha_S2 += pmodel.A.D' * alpha_kp' * Rinv * 
                                    (ey - m.Z * ex - alpha_kp * pmodel.A.f)
                end #if A

            end #if ZRUA

        end #for

        beta    = estimate_B ? beta_S1 \ beta_S2    : T[]
        nu      = estimate_U ? nu_S1 \ nu_S2        : T[]
        q       = estimate_Q ? q_S1 \ q_S2          : T[]

        zeta    = estimate_Z ? zeta_S1 \ zeta_S2    : T[]
        alpha   = estimate_A ? alpha_S1 \ alpha_S2  : T[]
        r       = estimate_R ? r_S1 \ r_S2          : T[]

        p       = length(params.x1) == 0 ? T[] : (pmodel.x1.D' * Linv * pmodel.x1.D) \
                        pmodel.x1.D' * Linv * (exs[:, 1] - pmodel.x1.f)
        lambda  = length(params.V1) == 0 ? T[] : (pmodel.V1.D' * pmodel.V1.D) \ pmodel.V1.D' *
                        vec(V[1] + exs[:, 1] * exs[:, 1]' -
                        exs[:, 1]*m.x1' - m.x1*exs[:, 1]' + m.x1*m.x1')

        toc()

        return SSMParameters(beta, nu, q, zeta, alpha, r, p, lambda), loglik

    end #em_kernel

    fraction_change(ll_prev, ll_curr) = isinf(ll_prev) ?
        1 : 2 * (ll_prev - ll_curr) / (ll_prev + ll_curr + 1e-6)

    ll, ll_prev = Inf, Inf

    while (niter > 0) & (fraction_change(ll_prev, ll) > eps)
        ll_prev     = ll 
        params, ll  = em_kernel(y, u, model, params)
        println(params)
        niter -= 1
    end #while

    niter > 0 ? nothing :
        warn("Parameter estimation timed out - results may have failed to converge")

    return params, model(params)
end #fit

function fit{T}(y::Array{T}, model::StateSpaceModel{T}; u::Array{T}=zeros(size(y,1), model.nu),
              eps::Float64=1e-6, niter::Int=typemax(Int))

    # B, Z, x1 default to parametrized as fully unconstrained
    B, B_params = parametrize_full(model.B)
    Z, Z_params = parametrize_full(model.Z)
    x1, x1_params = parametrize_full(model.x1)

    # U, A default to fixed
    U, U_params = parametrize_none(model.U)
    A, A_params = parametrize_none(model.A)

    # Q, R, V1 default to parametrized as diagonal with independent elements - any other values
    #   are ignored / set to zero 
    Q, Q_params = parametrize_diag(diag(model.Q))
    R, R_params = parametrize_diag(diag(model.R))
    V1, V1_params = parametrize_diag(diag(model.V1))

    pmodel = ParametrizedSSM(B, U, model.G, Q, Z, A, model.H, R, x1, V1)
    params = SSMParameters(B_params, U_params, Q_params, Z_params, A_params, R_params, x1_params, V1_params)
    fit(y, pmodel, params, u=u, eps=eps, niter=niter)

end #fit

