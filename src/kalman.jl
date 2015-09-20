type KalmanFiltered{T}
    y::Array{T}
    x::Array{T}
    V::Array{T, 3}
    x_pred::Array{T}
    V_pred::Array{T, 3}
    u::Array{T}
    K::Array{T, 3}
    model::StateSpaceModel{T}
    loglik::T
end

function show{T}(io::IO, filt::KalmanFiltered{T})
    n = size(filt.y, 1)
    dx, dy = filt.model.nx, filt.model.ny
    println("KalmanFiltered{$T}")
    println("$n observations, $dx-D process x $dy-D observations")
    println("Negative log-likelihood: $(filt.loglik)")
end

type KalmanSmoothed{T}
    y::Array{T}
    x::Array{T}
    V::Array{T, 3}
    V_lag1::Array{T, 3}
    u::Array{T}
    model::StateSpaceModel{T}
    loglik::T
end

function show{T}(io::IO, filt::KalmanSmoothed{T})
    n = size(filt.y, 1)
    dx, dy = filt.model.nx, filt.model.ny
    println("KalmanSmoothed{$T}")
    println("$n observations, $dx-D process x $dy-D observations")
    println("Negative log-likelihood: $(filt.loglik)")
end


function simulate(model::StateSpaceModel, n::Int; u::Array=zeros(n, model.nu))

    u = u'
    @assert size(u, 1) == model.nu
    @assert size(u, 2) == n

    # create empty arrays to hold the state and observed series
    x = zeros(model.nx, n)
    y = zeros(model.ny, n)

    # Cholesky decompositions of the covariance matrices, for generating
    # random noise
    Q_chol = @compat chol(model.Q, Val{:L})
    R_chol = @compat chol(model.R, Val{:L})

    # Generate the series
    x[:, 1] = model.x1
    y[:, 1] = model.Z * x[:, 1] + model.A * u[:,1] + R_chol * randn(model.ny)
    for i=2:n
        x[:, i] = model.B * x[:, i-1] + model.U * u[:,i-1] + Q_chol * randn(model.nx)
        y[:, i] = model.Z * x[:, i] + model.A * u[:,i] + R_chol * randn(model.ny)
    end

    return x', y'
end

function get_missing_value_filters(y::Vector)
    missing = isnan(y)
    select_present = eye(length(y))[find(!missing), :]
    select_missing = eye(length(y))[find(missing), :]
    return y.*!missing, select_present, select_missing
end #get_missing_value_filters

function process_missing_values(y::Vector, Z::Matrix, A::Matrix, R::Matrix)
    y, select_present, select_missing = get_missing_value_filters(y)
    keep_present = select_present' * select_present
    keep_missing = select_missing' * select_missing
    R_relevant = keep_present*R*keep_present + keep_missing*R*keep_missing
    return y, keep_present*Z, keep_present*A, R_relevant
end #function

function filter(y::Array, model::StateSpaceModel; u::Array=zeros(size(y,1), model.nu))

    y = y'
    u = u'
    n = size(y, 2)

    @assert !any(isnan(u)) #TODO: Better flexibility here 
    @assert n == size(u, 2)
    @assert model.ny == size(y, 1)
    @assert model.nu == size(u, 1)

    function predict_state(x_filt_prev::Vector, V_filt_prev::Matrix, u_prev::Vector, model::StateSpaceModel)
        x_pred_t = model.B * x_filt_prev + model.U * u_prev
        V_pred_t = model.B * V_filt_prev * model.B' + model.Q
        return x_pred_t, V_pred_t
    end #predict_state

    function filter_state(yt::Vector, x_pred_t::Vector, V_pred_t::Matrix, ut::Vector, model::StateSpaceModel)
        yt, Z, A, R = process_missing_values(yt, model.Z, model.A, model.R)
        innov     = yt - Z*x_pred_t - A*ut
        innov_cov = Z * V_pred_t * Z' + R
        Kt        = V_pred_t * Z' * inv(innov_cov)
        x_filt_t  = x_pred_t + Kt * innov
        V_filt_t  = (I - Kt * Z) * V_pred_t
        dll       = (dot(innov, innov_cov \ innov) + logdet(innov_cov))/2
        return x_filt_t, V_filt_t, Kt, dll
    end #filter_state

    x_pred = zeros(model.nx, n)
    V_pred = zeros(model.nx, model.nx, n)
    x_filt = zeros(x_pred)
    V_filt = zeros(model.nx, model.nx, n)
    K = zeros(model.nx, model.ny, n)
    log_likelihood = n*model.ny*log(2pi)/2

    x_pred[:, 1] = model.x1
    V_pred[:, :, 1] = model.V1
    x_filt[:, 1], V_filt[:, :, 1], K[:, :, 1], dll =
        filter_state(y[:, 1], x_pred[:, 1], V_pred[:, :, 1], u[:, 1], model)

    for t = 2:n
        x_pred[:, t], V_pred[:, :, t] =
            predict_state(x_filt[:, t-1], V_filt[:, :, t-1], u[:, t-1], model)
        x_filt[:, t], V_filt[:, :, t], K[:, :, t], dll =
            filter_state(y[:, t], x_pred[:, t], V_pred[:, :, t], u[:, t], model)
        log_likelihood += dll
    end #for

    return KalmanFiltered(y', x_filt', V_filt, x_pred', V_pred, u', K, model, log_likelihood)

end #filter

function smooth(y::Array, model::StateSpaceModel; u::Array=zeros(size(y,1), model.nu))

    filt = filter(y, model, u=u)

    n = size(y, 1)
    x_pred = filt.x_pred'
    x_filt = filt.x'
    x_smooth = zeros(model.nx, n)
    V_smooth = zeros(model.nx, model.nx, n)
    V_smooth_lag1 = zeros(model.nx, model.nx, n)
    J = zeros(model.nx, model.nx, n)

    Zn  = process_missing_values(collect(y[n, :]), model.Z, model.A, model.R)[2]

    for t = n:-1:2
        J[:, :, t]  = filt.V[:, :, t-1] * model.B' * inv(filt.V_pred[:, :, t])
    end #for
    J[:, :, 1]      = model.V1 * model.B' * inv(model.V1)

    x_smooth[:, n]          = x_filt[:, n]
    V_smooth[:, :, n]       = filt.V[:, :, n]
    V_smooth_lag1[:, :, n]  = (I - filt.K[:, :, n] * Zn) * model.B * filt.V[:, :, n-1]

    for t = (n-1):-1:1
        x_smooth[:, t]  = x_filt[:, t] + J[:, :, t+1] * (x_smooth[:, t+1] - x_pred[:, t+1])
        V_smooth[:, :, t]  =
            filt.V[:, :, t] + J[:, :, t+1] *
            (V_smooth[:, :, t+1] - filt.V_pred[:, :, t+1]) * J[:, :, t+1]'
        V_smooth_lag1[:, :, t] = filt.V[:, :, t] * J[:, :, t]' + J[:, :, t+1] *
            (V_smooth_lag1[:, :, t+1] - model.B * filt.V[:, :, t]) * J[:, :, t]'
    end #for


    return KalmanSmoothed(y, x_smooth', V_smooth, V_smooth_lag1, u, model, filt.loglik)

end #smooth
