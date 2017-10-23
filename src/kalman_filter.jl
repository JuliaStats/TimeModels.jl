type KalmanFiltered{T}
    filtered::Array{T}
    predicted::Array{T}
    error_cov::Array{T}
    pred_error_cov::Array{T}
    model::StateSpaceModel
    y::Array{T}
    u::Array{T}
    loglik::T
end

function Base.show{T}(io::IO, filt::KalmanFiltered{T})
    n = size(filt.y, 1)
    dx, dy = filt.model.nx, filt.model.ny
    println("KalmanFiltered{$T}")
    println("$n observations, $dx-D process x $dy-D observations")
    println("Negative log-likelihood: $(filt.loglik)")
end

function kalman_filter{T}(y::Array{T}, model::StateSpaceModel{T}; u::Array{T}=zeros(size(y,1), model.nu))

    @assert size(u,1) == size(y,1)
    @assert size(y,2) == model.ny
    @assert size(u,2) == model.nu

    function kalman_recursions(y_i::Vector{T}, u_i::Vector{T},
                                  C_i::Matrix{T}, D_i::Matrix{T}, W_i::Matrix{T},
                                  x_pred_i::Vector{T}, P_pred_i::Matrix{T})
        if !any(isnan, y_i)
            innov =  y_i - C_i * x_pred_i - D_i * u_i
            S = C_i * P_pred_i * C_i' + W_i  # Innovation covariance
            K = P_pred_i * C_i' / S # Kalman gain
            x_filt_i = x_pred_i + K * innov
            P_filt_i = (I - K * C_i) * P_pred_i
            dll = (dot(innov,S\innov) + logdet(S))/2
        else
            x_filt_i = x_pred_i
            P_filt_i = P_pred_i
            dll = 0
        end
        return x_filt_i, P_filt_i, dll
    end #kalman_recursions

    y = y'
    u = u'
    n = size(y, 2)
    x_pred = zeros(model.nx, n)
    x_filt = zeros(x_pred)
    P_pred = zeros(model.nx, model.nx, n)
    P_filt = zeros(P_pred)
    log_likelihood = n*model.ny*log(2pi)/2

    # first iteration
    A_1 = model.A(1)
    x_pred[:, 1] = model.x1
    P_pred[:, :, 1] = model.P1
    x_filt[:, 1], P_filt[:,:,1], dll = kalman_recursions(y[:, 1], u[:, 1],
                                            model.C(1), model.D(1), model.W(1),
                                            model.x1, model.P1)
    log_likelihood += dll

    for i=2:n
        A_i1 = model.A(i-1)
        x_pred[:, i] =  A_i1 * x_filt[:, i-1] + model.B(i-1) * u[:, i-1]
        P_pred[:, :, i] = A_i1 * P_filt[:, :, i-1] * A_i1' + model.V(i-1)
        x_filt[:, i], P_filt[:,:,i], dll = kalman_recursions(y[:, i], u[:, i],
                                                model.C(i), model.D(i), model.W(i),
                                                x_pred[:,i], P_pred[:,:,i])
        log_likelihood += dll
    end

    return KalmanFiltered(x_filt', x_pred', P_filt, P_pred, model, y', u', log_likelihood)
end

function loglikelihood{T}(y::Array{T}, model::StateSpaceModel{T}; u::Array{T}=zeros(size(y,1), model.nu))


    y, u = y', u'
    n = size(y, 2)

    @assert !any(isnan, u)
    @assert size(y, 1) == model.ny
    @assert size(u, 1) == model.nu
    @assert size(u, 2) == n

    y_notnan = .!isnan.(y)
    y = y .* y_notnan

    I0ny = zeros(model.ny, model.ny)
    Ksparse = issparse(model.A(1)) && issparse(model.V(1)) && issparse(model.C(1))

    x_pred_t, P_pred_t  = model.x1, model.P1
    ut, Ct, Wt          = zeros(model.nu), model.C(1), model.W(1)
    innov_t, innov_cov_inv_t  = zeros(model.ny), I0ny

    log_likelihood = n*model.ny*log(2pi)/2
    marginal_likelihood(innov::Vector, innov_cov::Matrix,
          innov_cov_inv::Matrix) =
              (dot(innov, innov_cov_inv * innov) + logdet(innov_cov))/2

    for t = 1:n

        # Predict using last iteration's values
        if t > 1

            At, But, Vt = model.A(t-1), model.B(t-1) * ut, model.V(t-1)
            AP_pred_t   = At * P_pred_t
            Kt          = AP_pred_t * Ct' * innov_cov_inv_t
            Ksparse && (Kt = sparse(Kt))

            x_pred_t = At * x_pred_t + But + Kt * innov_t
            P_pred_t = any(Wt .!= 0) ? AP_pred_t * (At - Kt * Ct)' + Vt : Vt
            P_pred_t = (P_pred_t + P_pred_t')/2

        end #if

        # Assign new values
        ut  = u[:, t]
        Ct  = model.C(t)
        Dut = model.D(t) * ut
        Wt  = model.W(t)

        # Check for and handle missing observation values
        missing_obs = !all(y_notnan[:, t])
        if missing_obs
            ynnt = y_notnan[:, t]
            I1, I2 = spdiagm(ynnt), spdiagm(!ynnt)
            Ct, Dut = I1 * Ct, I1 * Dut
            Wt = I1 * Wt * I1 + I2 * Wt * I2
        end #if

        # Compute nessecary filtering quantities
        innov_t           = y[:, t] - Ct * x_pred_t - Dut
        innov_cov_t       = Ct * P_pred_t * Ct' + Wt |> full
        nonzero_innov_cov = all(diag(innov_cov_t) .!= 0)
        innov_cov_inv_t   = nonzero_innov_cov ? inv(innov_cov_t) : I0ny

        nonzero_innov_cov && (log_likelihood += marginal_likelihood(innov_t, innov_cov_t, innov_cov_inv_t))

    end #for

    return log_likelihood

end #loglikelihood
