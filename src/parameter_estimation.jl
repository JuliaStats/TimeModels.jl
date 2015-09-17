# ML parameter estimation with filter result log-likelihood (via Nelder-Mead)

function fit{T}(y::Matrix{T}, build::Function, theta0::Vector{T};
          u::Array{T}=zeros(size(y,1), size(build(theta0).A, 2)))
    objective(theta) = filter(y, build(theta), u=u).loglik
    kfit = Optim.optimize(objective, theta0)
    return (kfit.minimum, build(kfit.minimum), filter(y, build(kfit.minimum)))
end #fit

# Expectation-Maximization (EM) parameter estimation

immutable Expectations{T}
    n::Int
    ey::Array{T, 2}
    eyy::Array{T, 3}
    ex::Array{T, 2}
    exx::Array{T, 3}
    eyx::Array{T, 3}
    eyx1::Array{T, 3}
    exx1::Array{T, 3}
    u::Array{T, 2}
    loglik::T
end #Expectations

function fit{T}(y::Array{T}, model::ParametrizedSSM, params::SSMParameters;
          u::Array{T}=zeros(size(y,1), size(model.A, 2)), eps::Float64=1e-6, niter::Int=typemax(Int))

    function expectation(y::Array, u::Array, m::StateSpaceModel)

      smoothed = smooth(y, m, u=u)
      y = y'
      u = u'

      n = size(y,2)
      ey = zeros(m.ny, n)
      eyy = zeros(m.ny, m.ny, n)
      ex = smoothed.x'
      exx = zeros(m.nx, m.nx, n)
      eyx = zeros(m.ny, m.nx, n)
      eyx1 = zeros(m.ny, m.nx, n-1)
      exx1 = zeros(m.nx, m.nx, n-1)

      for t  = 1:n
          yt, O1, O2 = get_missing_value_filters(y[:, t])
          Nt = I - m.R * O1' * inv(O1 * m.R * O1') * O1
          ey[:, t] = yt - Nt * (yt - m.Z * ex[:, t] - m.A * u[:, t])
          eyy[:, :, t] = O2'*O2 * (Nt * m.R + Nt * m.Z * smoothed.V[:, :, t] *
              m.Z' * Nt') * O2'*O2 + ey[:, t] * ey[:, t]'
          exx[:, :, t] = smoothed.V[:, :, t] + ex[:, t] * ex[:, t]'
          eyx[:, :, t] = Nt * m.Z * smoothed.V[:, :, t] + ey[:, t] * ex[:, t]'
      end #for

      for t = 1:n-1
          _, O1, _ = get_missing_value_filters(y[:, t+1])
          Nt = I - m.R * O1' * inv(O1 * m.R * O1') * O1
          eyx1[:, :, t] = Nt * m.Z * smoothed.V_lag1[:, :, t+1] + ey[:, t+1] * ex[:, t]'
          exx1[:, :, t] = smoothed.V_lag1[:, :, t+1] + ex[:, t+1] * ex[:, t]'
      end #for

      return Expectations(n, ey, eyy, ex, exx, eyx, eyx1, exx1, u, smoothed.loglik)
    end #expectation

    function maximization{T}(xp::Expectations, pmodel::ParametrizedSSM,
                params::SSMParameters{T})

        m = pmodel(params)

        # TODO: Premult u by U, A (and other common operations) for better performance?

        Qinv = inv(m.Q)
        Rinv = inv(m.R)
        Linv = inv(m.V1)

        C1(t) = kron(xp.u[:, t]', eye(m.nx))
        nu    = length(params.U) == 0 ? T[] : (inv(pmodel.U.D' * sum([(C1(t)' * Qinv * C1(t))''
                  for t in 2:xp.n]) * pmodel.U.D) *
                  sum([(C1(t)' * Qinv * (xp.ex[:, t] - m.B * xp.ex[:, t-1] - C1(t) * pmodel.U.f)
                  )'' for t in 2:xp.n]))[:]

        C2(t) = kron(xp.u[:, t]', eye(m.ny))
        alpha = length(params.A) == 0 ? T[] : (inv(pmodel.A.D' * sum([(C2(t)' * Rinv * C2(t))''
                  for t in 1:xp.n]) * pmodel.A.D) *
                  sum([(C2(t)' * Rinv * (xp.ey[:, t] - m.Z * xp.ex[:, t] - C2(t) * pmodel.A.f)
                  )'' for t in 1:xp.n]))[:]

        p     = length(params.x1) == 0 ? T[] : inv(pmodel.x1.D' * Linv * pmodel.x1.D) *
                        pmodel.x1.D' * Linv * (xp.ex[:, 1] - pmodel.x1.f)

        lambda = length(params.V1) == 0 ? T[] : inv(pmodel.V1.D' * pmodel.V1.D) * pmodel.V1.D' *
                        (xp.exx[:, :, 1] - xp.ex[:, 1]*m.x1' - m.x1*xp.ex[:, 1]' + m.x1*m.x1')[:]

        beta  = length(params.B) == 0 ? T[] : (inv(sum([(pmodel.B.D' * kron(xp.exx[:, :, t-1], Qinv) *
                  pmodel.B.D)'' for t=2:xp.n])) * sum([(pmodel.B.D' * ((Qinv *
                  xp.exx1[:, :, t-1])[:] - kron(xp.exx[:, :, t-1], Qinv) * pmodel.B.f -
                  (Qinv * m.U * xp.u[:, t] * xp.ex[:, t-1]')[:]))'' for t = 2:xp.n]))[:]

        zeta  = length(params.Z) == 0 ? T[] : (inv(sum([(pmodel.Z.D' * kron(xp.exx[:, :, t], Rinv) *
                  pmodel.Z.D)'' for t=1:xp.n])) * sum([(pmodel.Z.D' * ((Rinv * 
                  xp.eyx[:, :, t])[:] - kron(xp.exx[:, :, t], Rinv) * pmodel.Z.f -
                  (Rinv * m.A * xp.u[:, t] * xp.ex[:, t]')[:]))'' for t = 1:xp.n]))[:]


        q = length(params.Q) == 0 ? T[] : inv(sum([pmodel.Q.D' * pmodel.Q.D for t=2:xp.n])) *
                sum([pmodel.Q.D' * (xp.exx[:, :, t] - xp.exx1[:, :, t-1] * m.B' -
                m.B * xp.exx1[:, :, t-1]' - xp.ex[:, t] * xp.u[:, t]' * m.U'
                - m.U * xp.u[:, t] * xp.ex[:, t]' + m.B * xp.exx[:, :, t-1] * m.B' +
                m.B * xp.ex[:, t-1] * xp.u[:, t]' * m.U' +
                m.U * xp.u[:, t] * xp.ex[:, t-1]' * m.B' + m.U * xp.u[:, t] * xp.u[:, t]' * m.U')[:]
                for t = 2:xp.n])

        r = length(params.R) == 0 ? T[] : inv(sum([pmodel.R.D' * pmodel.R.D for t=1:xp.n])) *
                sum([pmodel.R.D' * (xp.eyy[:, :, t] - xp.eyx[:, :, t] * m.Z' -
                m.Z * xp.eyx[:, :, t]' - xp.ey[:, t] * xp.u[:, t]' * m.A' -
                m.A * xp.u[:, t] * xp.ey[:, t]' + m.Z * xp.exx[:, :, t] * m.Z' +
                m.Z * xp.ex[:, t] * xp.u[:, t]' * m.A' + m.A * xp.u[:, t] * xp.ex[:, t]' * m.Z' +
                m.A * xp.u[:, t] * xp.u[:, t]' * m.A')[:]
                for t = 1:xp.n])

        return SSMParameters(beta, nu, q, zeta, alpha, r, p, lambda)

    end #maximize

    fraction_change(ll_prev, ll_curr) = isinf(ll_prev) ?
        1 : 2 * (ll_prev - ll_curr) / (ll_prev + ll_curr + 1e-6)

    ll_prev       = Inf
    expectations  = expectation(y, u, model(params))
    params        = maximization(expectations, model, params)
    niter -= 1

    while (niter > 0) & (fraction_change(ll_prev, expectations.loglik) > eps)
        ll_prev       = expectations.loglik
        expectations  = expectation(y, u, model(params))
        params        = maximization(expectations, model, params)
        niter -= 1
    end #while

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

    # Q, R, V1 default to parametrized as diagonal with independent elements - any other values are set to constants
    Q, Q_params = parametrize_diag(model.Q)
    R, R_params = parametrize_diag(model.R)
    V1, V1_params = parametrize_diag(model.V1)

    pmodel = ParametrizedSSM(B, U, Q, Z, A, R, x1, V1)
    params = SSMParameters(B_params, U_params, Q_params, Z_params, A_params, R_params, x1_params, V1_params)
    fit(y, pmodel, params, u=u, eps=eps, niter=niter)

end #fit

