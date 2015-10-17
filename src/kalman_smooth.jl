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

function show{T}(io::IO, smoothed::KalmanSmoothed{T})
    n = size(smoothed.y, 1)
    dx, dy = smoothed.model.nx, smoothed.model.ny
    println("KalmanSmoothed{$T}")
    println("$n observations, $dx-D process x $dy-D observations")
    println("Negative log-likelihood: $(smoothed.loglik)")
end

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
        J = P_filt[:, :, i] * model.F(i)' * inv(P_pred[:,:,i+1])
        x_smooth[:, i] = x_filt[:, i] + J * (x_smooth[:, i+1] - x_pred[:, i+1])
        P_smoov[:, :, i] = P_filt[:, :, i] + J * (P_smoov[:, :, i+1] - P_pred[:,:,i+1]) * J'
    end

    return KalmanSmoothed(x_pred', x_filt', x_smooth', P_smoov, model, filt.y, filt.u, filt.loglik)
end

kalman_smooth(y::Array, model::StateSpaceModel; u=zeros(size(y,1), model.nu)) =
    kalman_filter(y, model, u=u) |> kalman_smooth
