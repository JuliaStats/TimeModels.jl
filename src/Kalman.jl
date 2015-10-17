type KalmanFiltered{T}
	filtered::Array{T}
	predicted::Array{T}
	error_cov::Array{T}
	pred_error_cov::Array{T}
	model::StateSpaceModel
	y::Array{T}
	loglik::T
end

function show{T}(io::IO, filt::KalmanFiltered{T})
	n = size(filt.y, 1)
	dx, dy = length(filt.model.x0), size(filt.model.G, 1)
	println("KalmanFiltered{$T}")
	println("$n observations, $dx-D process x $dy-D observations")
	println("Negative log-likelihood: $(filt.loglik)")
end

type KalmanSmoothed{T}
	predicted::Array{T}
	filtered::Array{T}
	smoothed::Array{T}
	error_cov::Array{T}
	model::StateSpaceModel
	y::Array{T}
	loglik::T
end

function show{T}(io::IO, filt::KalmanSmoothed{T})
	n = size(filt.y, 1)
	dx, dy = length(filt.model.x0), size(filt.model.G, 1)
	println("KalmanSmoothed{$T}")
	println("$n observations, $dx-D process x $dy-D observations")
	println("Negative log-likelihood: $(filt.loglik)")
end

function eval_matrix(M::Matrix{Function}, i::Int, ArrayType::Type)
    reshape(convert(Array{ArrayType}, [m(i) for m in M]), size(M))
end
function eval_matrix{T <: Real}(M::Matrix{T}, n::Int, ArrayType::Type)
    M
end

function simulate{T}(model::StateSpaceModel{T}, n::Int)
    # Generates a realization of a state space model.
    #
    # Arguments:
    # model : StateSpaceModel
    #	Model defining the process
    # n : Int
    #	Number of steps to simulate.

    # dimensions of the process and observation series
    nx = length(model.x0)
    ny = size(model.G, 1)

    # create empty arrays to hold the state and observed series
    x = zeros(nx, n)
    y = zeros(ny, n)

    # Cholesky decompositions of the covariance matrices, for generating
    # random noise
    V_chol = @compat chol(model.V, Val{:L})
    W_chol = @compat chol(model.W, Val{:L})

    # Generate the series
    x[:, 1] = eval_matrix(model.F, 1, T) * model.x0 + V_chol * randn(nx)
    y[:, 1] = eval_matrix(model.G, 1, T) * x[:, 1] + W_chol * randn(ny)
    for i=2:n
        x[:, i] = eval_matrix(model.F, i, T) * x[:, i-1] + V_chol * randn(nx)
        y[:, i] = eval_matrix(model.G, i, T) * x[:, i] + W_chol * randn(ny)
    end

    return x', y'
end

function kalman_filter{T}(y::Array{T}, model::StateSpaceModel{T})
    @assert size(y,2) == size(model.G,1)

    function kalman_recursions(y_i, G_i, W, x_pred_i, P_pred_i)
        if !any(isnan(y_i))
            innov =  y_i - G_i * x_pred_i
            S = G_i * P_pred_i * G_i' + W  # Innovation covariance
            K = P_pred_i * G_i' / S #* inv(S)	# Kalman gain
            x_filt_i = x_pred_i + K * innov
            P_filt_i = (I - K * G_i) * P_pred_i
            dll = (dot(innov,S\innov) + logdet(S))/2
        else
            x_filt_i = x_pred_i
            P_filt_i = P_pred_i
            dll = 0
        end
        return x_filt_i, P_filt_i, dll
    end #kalman_recursions

    y = y'
    ny = size(y,1)
    n = size(y, 2)
    x_pred = zeros(length(model.x0), n)
    x_filt = zeros(x_pred)
    P_pred = zeros(size(model.P0, 1), size(model.P0, 2), n)
    P_filt = zeros(P_pred)
    log_likelihood = n*ny*log(2pi)/2

    # first iteration
    F_1 = eval_matrix(model.F, 1, T)
    x_pred[:, 1] = F_1 * model.x0
    P_pred[:, :, 1] = F_1 * model.P0 * F_1' + model.V
    x_filt[:, 1], P_filt[:,:,1], dll = kalman_recursions(y[:, 1], eval_matrix(model.G, 1, T),
    model.W, x_pred[:,1], P_pred[:,:,1])
    log_likelihood += dll

    for i=2:n
        F_i = eval_matrix(model.F, i, T)
        x_pred[:, i] =  F_i * x_filt[:, i-1]
        P_pred[:, :, i] = F_i * P_filt[:, :, i-1] * F_i' + model.V
        x_filt[:, i], P_filt[:,:,i], dll = kalman_recursions(y[:, i], eval_matrix(model.G, i, T),
        model.W, x_pred[:,i], P_pred[:,:,i])
        log_likelihood += dll
    end

    return KalmanFiltered(x_filt', x_pred', P_filt, P_pred, model, y', log_likelihood)
end


function kalman_smooth{T}(y::Array{T}, model::StateSpaceModel{T}; filt::KalmanFiltered{T} = kalman_filter(y, model))
    n = size(y, 1)
    x_pred = filt.predicted'
    x_filt = filt.filtered'
    x_smooth = zeros(size(x_filt))
    P_pred = filt.pred_error_cov
    P_filt = filt.error_cov
    P_smoov = zeros(P_filt)

    x_smooth[:, n] = x_filt[:, n]
    P_smoov[:, :, n] = P_filt[:, :, n]
    for i = (n-1):-1:1
        J = P_filt[:, :, i] * eval_matrix(model.F, i, T)' * inv(P_pred[:,:,i+1])
        x_smooth[:, i] = x_filt[:, i] + J * (x_smooth[:, i+1] - x_pred[:, i+1])
        P_smoov[:, :, i] = P_filt[:, :, i] + J * (P_smoov[:, :, i+1] - P_pred[:,:,i+1]) * J'
    end

    return KalmanSmoothed(x_pred', x_filt', x_smooth', P_smoov, model, y, filt.loglik)
end

function kalman_smooth{T}(kfiltered::KalmanFiltered{T})
    kalman_smooth(kfiltered.y, kfiltered.model, filt = kfiltered)
end


function fit{T}(y::Matrix{T}, build::Function, theta0::Vector{T})
    objective(theta) = kalman_filter(y, build(theta)).loglik
    kfit = Optim.optimize(objective, theta0)
    return (kfit.minimum, build(kfit.minimum))
end
