type KalmanSmoothed{T}
	predicted::Array{T}
	filtered::Array{T}
	smoothed::Array{T}
	error_cov::Array{T}
	model::StateSpaceModel{T}
	y::Array{T}
  u::Array{T}
	loglik::T
end

type KalmanSmoothedMinimal{T}
	y::Array{T}
	smoothed::Array{T}
	error_cov::Union{Vector{Matrix{T}}, Vector{SparseMatrixCSC{T, Int}}}
  u::Array{T}
	model::StateSpaceModel{T}
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
function kalman_smooth{T}(y::Array{T}, model::StateSpaceModel{T}; u::Array{T}=zeros(size(y,1), model.nu))

    Ksparse = issparse(model.A(1)) && issparse(model.V(1)) && issparse(model.C(1))

    y_orig = copy(y)
    y = y'
    u_orig = copy(u)
    u = u'
    n = size(y, 2)

    @assert !any(isnan, u)
    # @assert size(y, 1) == model.ny
    if size(y, 1) != model.ny
        @show size(y)
        @show model.ny
        error("")
    end
    @assert size(u, 1) == model.nu
    @assert size(u, 2) == n

    I0ny = zeros(model.ny, model.ny)

    y_notnan = (!).(isnan.(y))
    y = y .* y_notnan

    x_pred_t, P_pred_t  = model.x1, model.P1
    x, P                = zeros(model.nx, n), Array{typeof(P_pred_t)}(0)

    innov_t             = zeros(model.ny)
    innov_cov_t         = zeros(model.ny, model.ny)
    innov_cov_inv_t     = copy(innov_cov_t)
    Kt                  = Ksparse ? spzeros(model.nx, model.ny) : zeros(model.nx, model.ny)
    KCt                 = Ksparse ? spzeros(model.nx, model.nx) : zeros(model.nx, model.nx)

    innov               = zeros(model.ny, n)
    innov_cov_inv       = zeros(model.ny, model.ny, n)
    KC                  = Array{typeof(KCt)}(0)

    ut, Ct, Wt     = zeros(model.nu), model.C(1), model.W(1)

    log_likelihood = n*model.ny*log(2pi)/2
    marginal_likelihood(innov::Vector, innov_cov::Matrix,
          innov_cov_inv::Matrix) =
              (dot(innov, innov_cov_inv * innov) + logdet(innov_cov))/2

    # Prediction and filtering
    for t = 1:n

        # Predict using last iteration's values
        if t > 1

            At, But, Vt = model.A(t-1), model.B(t-1) * ut, model.V(t-1)
            AP_pred_t   = At * P_pred_t
            Kt          = AP_pred_t * Ct' * innov_cov_inv_t
            Ksparse && (Kt = sparse(Kt))
            KCt         = Kt * Ct
            push!(KC, KCt)

            x_pred_t = At * x_pred_t + But + Kt * innov_t
            P_pred_t = any(Wt .!= 0) ? AP_pred_t * (At - KCt)' + Vt : Vt

        end #if
        x[:, t]  = x_pred_t
        push!(P, (P_pred_t + P_pred_t')/2)

        # Assign new values
        ut  = u[:, t]
        Ct  = model.C(t)
        Dut = model.D(t) * ut
        Wt  = model.W(t)

        # Check for and handle missing observation values
        missing_obs = !all(y_notnan[:, t])
        if missing_obs
            ynnt = y_notnan[:, t]
            I1, I2 = spdiagm(ynnt), spdiagm(.!ynnt)
            Ct, Dut = I1 * Ct, I1 * Dut
            Wt = I1 * Wt * I1 + I2 * Wt * I2
        end #if

        # Compute nessecary filtering quantities
        innov_t           = y[:, t] - Ct * x_pred_t - Dut
        innov_cov_t       = Ct * P_pred_t * Ct' + Wt |> full
        nonzero_innov_cov = all(diag(innov_cov_t) .!= 0)
        innov_cov_inv_t   = nonzero_innov_cov ? inv(innov_cov_t) : I0ny

        innov[:, t]             = innov_t
        innov_cov_inv[:, :, t]  = innov_cov_inv_t

        nonzero_innov_cov && (log_likelihood += marginal_likelihood(innov_t, innov_cov_t, innov_cov_inv_t))

    end #for
    push!(KC, zeros(KC[1]))

    # Smoothing
    r = zeros(model.nx)
    N = zeros(model.P1)

    for t = n:-1:1

        Ct = !all(y_notnan[:, t]) ? spdiagm(y_notnan[:, t]) * model.C(t) : model.C(t)

        CF = Ksparse ? sparse(Ct' * innov_cov_inv[:, :, t]) : Ct' * innov_cov_inv[:, :, t]
        P_pred_t = P[t]
        L = model.A(t) - KC[t]

        r = CF * innov[:, t] + L' * r
        N = CF * Ct  + L' * N * L

        x[:, t] = x[:, t] + P_pred_t * r
        P_smooth_t = P_pred_t  - P_pred_t * N * P_pred_t
        P[t] = (P_smooth_t + P_smooth_t')/2

    end #for

    return KalmanSmoothedMinimal(y_orig, x', P, u_orig, model, log_likelihood)

end #smooth

function lag1_smooth{T}(y::Array{T}, u::Array{T}, m::StateSpaceModel{T})

    A_0, A_I  = zeros(m.A(1)), eye(m.A(1))
    B_0, V_0, C_0 = zeros(m.B(1)), zeros(m.V(1)), zeros(m.C(1))
    x1_0, P1_0  = zeros(m.x1), zeros(m.P1)

    A_stack(t) = [m.A(t) A_0; A_I A_0]
    B_stack(t) = [m.B(t); B_0]
    V_stack(t) = [m.V(t) V_0; V_0 V_0]
    C_stack(t) = [m.C(t) C_0]
    x1_stack   = [m.x1; x1_0]
    P1_stack   = [m.P1 P1_0; P1_0 P1_0]
    stack_model = StateSpaceModel(A_stack, V_stack, C_stack, m.W,
                                      x1_stack, P1_stack, B=B_stack, D=m.D)
    stack_smoothed = kalman_smooth(y, stack_model, u=u)

    x     = stack_smoothed.smoothed[:, 1:m.nx]'

    err_cov_type = typeof(m.P1)
    P     = err_cov_type[P[1:m.nx, 1:m.nx] for P in stack_smoothed.error_cov]
    Plag1 = err_cov_type[P[1:m.nx, (m.nx+1):end] for P in stack_smoothed.error_cov]

    return x, P, Plag1, stack_smoothed.loglik

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

