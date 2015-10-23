type KalmanSmoothed{T}
	predicted::Array{T}
	filtered::Array{T}
	smoothed::Array{T}
	error_cov::Array{T}
	model::StateSpaceModel
	y::Array{T}
  u::Array{T}
	loglik::T
end

type KalmanSmoothedMinimal{T}
	y::Array{T}
	smoothed::Array{T}
	error_cov::Array{T}
  u::Array{T}
	model::StateSpaceModel
	loglik::T
end

function show{T}(io::IO, smoothed::KalmanSmoothed{T})
    n = size(smoothed.y, 1)
    dx, dy = smoothed.model.nx, smoothed.model.ny
    println("KalmanSmoothed{$T}")
    println("$n observations, $dx-D process x $dy-D observations")
    println("Negative log-likelihood: $(smoothed.loglik)")
end

function show{T}(io::IO, smoothed::KalmanSmoothedMinimal{T})
    n = size(smoothed.y, 1)
    dx, dy = smoothed.model.nx, smoothed.model.ny
    println("KalmanSmoothedMinimal{$T}")
    println("$n observations, $dx-D process x $dy-D observations")
    println("Negative log-likelihood: $(smoothed.loglik)")
end

# Durbin-Koopman Smoothing
function kalman_smooth(y::Array, model::StateSpaceModel; u::Array=zeros(size(y,1), model.nu))

    y_orig = copy(y)
    y = y'
    u = u'
    n = size(y, 2)

    @assert !any(isnan(u))
    @assert size(y, 1) == model.ny
    @assert size(u, 1) == model.nu
    @assert size(u, 2) == n

    y_notnan = !isnan(y)
    y = y .* y_notnan

    x_pred,  V_pred     = zeros(model.nx, n), zeros(model.nx, model.nx, n)
    x_smooth, V_smooth  = copy(x_pred), copy(V_pred)
    x_pred_t, V_pred_t  = model.x1, model.P1

    innov               = zeros(model.ny, n)
    innov_cov_inv       = zeros(model.ny, model.ny, n)
    K                   = zeros(model.nx, model.ny, n)

    innov_t             = zeros(model.ny)
    innov_cov_t         = zeros(model.ny, model.ny)
    innov_cov_inv_t     = copy(innov_cov_t)
    Kt                  = zeros(model.nx, model.ny)

    At, Ct, Dt, Wt, ut  = model.A(1), model.C(1), model.D(1), model.W(1), zeros(model.nu)

    log_likelihood = n*model.ny*log(2pi)/2
    marginal_likelihood(innov::Vector, innov_cov::Matrix, innov_cov_inv::Matrix) =
         (dot(innov, innov_cov_inv * innov) + logdet(innov_cov))/2

    # Prediction and filtering
    for t = 1:n

        # Predict using last iteration's values
        if t > 1
            x_pred_t = At * x_pred_t + model.B(t-1) * ut + Kt * innov_t
            V_pred_t = At * V_pred_t * (At - Kt * Ct)' + model.V(t)
            V_pred_t = (V_pred_t + V_pred_t')/2
        end #if
        x_pred[:, t]    = x_pred_t
        V_pred[:, :, t] = V_pred_t

        # Assign new values
        At = model.A(t)
        Ct = model.C(t)
        Dt = model.D(t)
        Wt = model.W(t)
        ut = u[:, t]

        # Check for and handle missing observation values
        missing_obs = !all(y_notnan[:, t])
        if missing_obs
            ynnt = y_notnan[:, t]
            I1, I2 = diagm(ynnt), diagm(!ynnt)
            Ct, Dt = I1 * Ct, I1 * Dt
            Wt = I1 * Wt * I1 + I2 * Wt * I2
        end #if

        # Compute nessecary filtering quantities
        innov_t         = y[:, t] - Ct * x_pred_t - Dt * ut
        innov_cov_t     = Ct * V_pred_t * Ct' + Wt
        innov_cov_inv_t = inv(innov_cov_t)
        Kt              = At * V_pred_t * Ct' * innov_cov_inv_t

        innov[:, t]             = innov_t
        innov_cov_inv[:, :, t]  = innov_cov_inv_t
        K[:, :, t]              = Kt
        log_likelihood += marginal_likelihood(innov_t, innov_cov_t, innov_cov_inv_t)

    end #for

    # Smoothing
    r, N = zeros(model.nx), zeros(model.nx, model.nx)

    for t = n:-1:1

        Ct = model.C(t)
        innov_cov_inv_t = innov_cov_inv[:, :, t]
        V_pred_t = V_pred[:, :, t]
        L = model.A(t) - K[:, :, t] * Ct

        r = Ct' * innov_cov_inv_t * innov[:, t] + L' * r
        N = Ct' * innov_cov_inv_t * Ct + L' * N * L

        x_smooth[:, t] = x_pred[:, t] + V_pred_t * r
        V_smooth_t = V_pred_t - V_pred_t * N * V_pred_t
        V_smooth[:, :, t] =  (V_smooth_t + V_smooth_t')/2

    end #for

    return KalmanSmoothedMinimal(y_orig, x_smooth', V_smooth, u', model, log_likelihood)

end #smooth

# Bad performance - will be much better with sparse matrices
function lag1_smooth(y::Array, u::Array, m::StateSpaceModel)

    A_stack(t) = [m.A(t) zeros(m.nx, m.nx); eye(m.nx) zeros(m.nx, m.nx)]
    B_stack(t) = [m.B(t); zeros(m.nx, m.nu)]
    V_stack(t) = [m.V(t) zeros(m.nx, m.nx); zeros(m.nx, 2*m.nx)]
    C_stack(t) = [m.C(t) zeros(m.ny, m.nx)]
    x1_stack   = [m.x1; zeros(model.x1)]
    P1_stack   = [m.P1 zeros(m.nx, m.nx); zeros(m.nx, 2*m.nx)]
    stack_model = StateSpaceModel(A_stack, B_stack, V_stack,
                                  C_stack, model.D, model.W, x1_stack, V1_stack)

    stack_smoothed = smooth(y, stack_model, u=u)
    x     = stack_smoothed.x[:, 1:model.nx]
    V     = stack_smoothed.V[1:model.nx, 1:model.nx, :]
    Vlag1 = stack_smoothed.V[1:model.nx, (model.nx+1):end]
    return x, V, Vlag1, stack_smoothed.loglik

end #function


# Rauch-Tung-Striebel Smoothing
function kalman_smooth(filt::KalmanFiltered)

    n = size(filt.y, 1)
    model = filt.model
    x_pred = filt.predicted'
    x_filt = filt.filtered'
    x_smooth = zeros(x_filt)
    P_pred = filt.pred_error_cov
    P_filt = filt.error_cov
    P_smoov = zeros(P_filt)

    x_smooth[:, n] = x_filt[:, n]
    P_smoov[:, :, n] = P_filt[:, :, n]
    for i = (n-1):-1:1
        J = P_filt[:, :, i] * model.A(i)' * inv(P_pred[:,:,i+1])
        x_smooth[:, i] = x_filt[:, i] + J * (x_smooth[:, i+1] - x_pred[:, i+1])
        P_smoov[:, :, i] = P_filt[:, :, i] + J * (P_smoov[:, :, i+1] - P_pred[:,:,i+1]) * J'
    end

    return KalmanSmoothed(x_pred', x_filt', x_smooth', P_smoov, model, filt.y, filt.u, filt.loglik)
end

