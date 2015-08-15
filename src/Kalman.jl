issquare(x::Matrix) = size(x, 1) == size(x, 2) ? true : false

function check_dimensions(F, V, G, W, x0, P0)
	@assert length(x0) == size(F, 2)
	@assert size(F) == size(V)
	@assert issquare(F)
	@assert size(G, 2) == size(F, 1)
	@assert size(G, 1) == size(W, 1)
	@assert issquare(W)
	@assert size(P0, 1) == length(x0)
	@assert issquare(P0)
end

type StateSpaceModel{T}
	# Process transition and noise covariance
	F::Union(Matrix{T}, Matrix{Function})
	V::Matrix{T}
	# Observation and noise covariance
	G::Union(Matrix{T}, Matrix{Function})
	W::Matrix{T}
	# Inital guesses at state and error covariance
	x0::Vector{T}
	P0::Matrix{T}

	function StateSpaceModel(F::Union(Matrix{T}, Matrix{Function}), V::Matrix{T},
	                G::Union(Matrix{T}, Matrix{Function}), W::Matrix{T}, x0::Vector{T}, P0::Matrix{T})
		check_dimensions(F, V, G, W, x0, P0)
		new(F, V, G, W, x0, P0)
	end
end

function StateSpaceModel{T <: Real}(F::Union(Matrix{T}, Matrix{Function}), V::Matrix{T}, G::Union(Matrix{T}, Matrix{Function}),
		W::Matrix{T}, x0::Vector{T}, P0::Matrix{T})
	StateSpaceModel{T}(F, V, G, W, x0, P0)
end

function show{T}(io::IO, mod::StateSpaceModel{T})
	dx, dy = length(mod.x0), size(mod.G, 1)
	println("StateSpaceModel{$T}, $dx-D process x $dy-D observations")
	println("Process evolution matrix F:")
	show(mod.F)
	println("\n\nProcess error covariance V:")
	show(mod.V)
	println("\n\nObservation matrix G:")
	show(mod.G)
	println("\n\nObseration error covariance W:")
	show(mod.W)
end

type KalmanFiltered{T}
	filtered::Array{T}
	predicted::Array{T}
	error_cov::Array{T}
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
	filtered::Array{T}
	predicted::Array{T}
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
	x[:, 1] = model.x0
	y = zeros(ny, n)
        y[:, 1] = eval_matrix(model.G, 1, T) * x[:, 1]
	# Cholesky decompositions of the covariance matrices, for generating
	# random noise
	V_chol = chol(model.V)'
	W_chol = chol(model.W)'
	# Generate the series
        for i=2:n
            x[:, i] = eval_matrix(model.F, i, T) * x[:, i-1] + V_chol * randn(nx)
            y[:, i] = eval_matrix(model.G, i, T) * x[:, i] + W_chol * randn(ny)
        end
	return x', y'
end

function loglik{T}(innov::Array{T}, S::Array{T})
	log(1) - 0.5log(2pi*det(S)) + 0.5log(det(S)) - 0.5*innov' * S * innov
end

function kalman_filter{T}(y::Array{T}, model::StateSpaceModel{T})
	n = size(y, 1)
	y = y'
	x_filt = zeros(length(model.x0), n)
	x_pred = zeros(size(x_filt))
	P = zeros(size(model.P0, 1), size(model.P0, 2), n)
	P[:, :, 1] = model.P0
	P0 = model.P0
	I = eye(size(x_filt, 1))
	# first iteration
	x_filt[:, 1] = eval_matrix(model.F, 1, T) * model.x0
	P[:, :, 1] = eval_matrix(model.F, 1, T) * P0 * eval_matrix(model.F, 1, T)' + model.V
	innovation =  y[:, 1] - eval_matrix(model.G, 1, T) * x_filt[:, 1]
	S = eval_matrix(model.G, 1, T) * P[:, :, 1] * eval_matrix(model.G, 1, T)' + model.W   # Innovation covariance
	K = P[:, :, 1] * eval_matrix(model.G, 1, T)' * inv(S)				      # Kalman gain
	x_pred[:, 1] = x_filt[:, 1]
	x_filt[:, 1] = x_filt[:, 1] + K * innovation
	P[:, :, 1] = (I - K * eval_matrix(model.G, 1, T)) * P[:, :, 1]
	log_likelihood = 0
	for i=2:n
		# prediction
		x_filt[:, i] = eval_matrix(model.F, i, T) * x_filt[:, i-1]
		P[:, :, i] = eval_matrix(model.F, i, T) * P[:, :, i-1] * eval_matrix(model.F, i, T)' + model.V
		# evaluate the likelihood
		innovation =  y[:, i] - eval_matrix(model.G, i, T) * x_filt[:, i]
		S = eval_matrix(model.G, i, T) * P[:, :, i] * eval_matrix(model.G, i, T)' + model.W  # Innovation covariance
		log_likelihood -= loglik(innovation, S)
		# update
		K = P[:, :, i] * eval_matrix(model.G, i, T)' * inv(S)				     # Kalman gain
		x_pred[:, i] = x_filt[:, i]
		x_filt[:, i] = x_filt[:, i] + K * innovation
		P[:, :, i] = (I - K * eval_matrix(model.G, i, T)) * P[:, :, i]
	end
	return KalmanFiltered(x_filt', x_pred', P, model, y', log_likelihood[1])
end


function kalman_smooth{T}(y::Array{T}, model::StateSpaceModel{T})
	filt = kalman_filter(y, model)
	n = size(y, 1)
	x_filt = filt.filtered'
	x_pred = filt.predicted'
	x_smooth = zeros(size(x_filt))
	P = filt.error_cov
	P_smoov = zeros(size(P))
	model = filt.model

	P_pred = eval_matrix(model.F, n, T) * P[:, :, n-1] * eval_matrix(model.F, n, T)' + model.V
	J = P[:, :, n] * eval_matrix(model.F, n, T)' * inv(P_pred)
	x_smooth[:, n] = x_filt[:, n]
	for i = (n-1):-1:1
		J = P[:, :, i] * eval_matrix(model.F, i, T)' * inv(P_pred)
		x_smooth[:, i] = x_filt[:, i] + J * (x_smooth[:, i+1] - x_pred[:, i+1])
		P_smoov[:, :, i] = P[:, :, i] * J * (P_smoov[:, :, i+1] - P_pred) * J'
	end
	P_pred = eval_matrix(model.F, 1, T) * model.P0 * eval_matrix(model.F, 1, T)' + model.V
	J = P[:, :, 1] * eval_matrix(model.F, 1, T)' * inv(P_pred)
	x_smooth[:, 1] = x_filt[:, 1] + J * (x_smooth[:, 2] - x_pred[:, 2])
	P_smoov[:, :, 1] = P[:, :, 1] * J * (P_smoov[:, :, 2] - P_pred) * J'

	return KalmanSmoothed(x_filt', x_pred', x_smooth, P_smoov[:,:,end],
		model, y, filt.loglik)
end

function fit{T}(y::Array{T}, build::Function, theta0::Vector{T})
	objective(theta) = kalman_filter(y, build(theta)).loglik
	kfit = Optim.optimize(objective, theta0)
	return (kfit.minimum, build(kfit.minimum))
end
