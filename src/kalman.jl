using Rmath

type StateSpaceModel{T}
	# Process transition and noise covariance
	F::Array{T}
	V::Array{T}
	# Observation and noise covariance
	G::Array{T}
	W::Array{T}
	# Inital guesses at state and error covariance
	x0::Array{T}
	P0::Array{T}
end

type KalmanFiltered{T}
	filtered::Array{T}
	predicted::Array{T}
	error_cov::Array{T}
	model::StateSpaceModel
	y::Array{T}
	loglik::T
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

function issquare(x::Array)
	return size(x, 1) == size(x, 2)
end

function simulate_statespace{T}(n::Int, model::StateSpaceModel{T})
	# Generates a realization of a state space model.
	#
	# Arguments:
	# n : Int
	#	Number of steps to simulate.
	# model : StateSpaceModel
	#	Model defining the process
	# x0 : Array{float}.
	#	Inital state vector.

	# dimensions of the process and observation series
	nx = length(model.x0)
	ny = size(model.G, 1)
	# create empty arrays to hold the state and observed series
	x = zeros(nx, n)
	x[:, 1] = model.x0
	y = zeros(ny, n)
	y[:, 1] = model.G * x[:, 1]
	# Cholesky decompositions of the covariance matrices, for generating
	# random noise
	V_chol = chol(model.V)'
	W_chol = chol(model.W)'
	# Generate the series
	for i=2:n
		x[:, i] = model.F * x[:, i-1] + V_chol * rnorm(nx)
		y[:, i] = model.G * x[:, i] + W_chol * rnorm(ny)
	end
	return x', y'
end

function check_dimensions(model::StateSpaceModel)
	@assert ndims(model.F) == ndims(model.G) == 2
	@assert ndims(model.V) == ndims(model.W) == ndims(model.P0) == 2
	@assert length(model.x0) == size(model.F, 2)
	@assert size(model.F) == size(model.V)
	@assert issquare(model.F)
	@assert size(model.G, 2) == size(model.F, 1)
	@assert size(model.G, 1) == size(model.W, 1)
	@assert issquare(model.W)
	@assert size(model.P0, 1) == length(model.x0)
	@assert issquare(model.P0)
end

function kalman_filter{T}(y::Array{T}, model::StateSpaceModel{T})
	check_dimensions(model)
	n = size(y, 1)
	y = y'
	x_filt = zeros(length(model.x0), n)
	x_pred = zeros(size(x_filt))
	P = zeros(size(model.P0, 1), size(model.P0, 2), n)
	P[:, :, 1] = model.P0
	P0 = model.P0
	I = eye(size(x_filt, 1))

	x_filt[:, 1] = model.F * model.x0
	P[:, :, 1] = model.F * P0 * model.F' + model.V
	innovation =  y[:, 1] - model.G * x_filt[:, 1]
	S = model.G * P[:, :, 1] * model.G' + model.W 	# Innovation covariance
	K = P[:, :, 1] * model.G' * inv(S)				# Kalman gain
	x_pred[:, 1] = x_filt[:, 1]
	x_filt[:, 1] = x_filt[:, 1] + K * innovation
	P[:, :, 1] = (I - K * model.G) * P[:, :, 1]
	log_likelihood = 0
	for i=2:n
		# prediction
		x_filt[:, i] = model.F * x_filt[:, i-1]
		P[:, :, i] = model.F * P[:, :, i-1] * model.F' + model.V
		# evaluate the likelihood
		innovation =  y[:, i] - model.G * x_filt[:, i]
		sigma = model.G * P[:, :, i] * model.G' + model.W
		innovation =  y[:, i] - model.G * x_filt[:, i]
		py = 1 / (sqrt(2pi) * det(sigma)^0.5) * exp(-0.5 * innovation' * sigma * innovation)
		log_likelihood += log(py)
		# update
		S = model.G * P[:, :, i] * model.G' + model.W 	# Innovation covariance
		K = P[:, :, i] * model.G' * inv(S)				# Kalman gain
		x_pred[:, i] = x_filt[:, i]
		x_filt[:, i] = x_filt[:, i] + K * innovation
		P[:, :, i] = (I - K * model.G) * P[:, :, i]

	end
	return KalmanFiltered(x_filt', x_pred', P, model, y', log_likelihood[1])
end

function kalman_smooth{T}(y::Array{T}, model::StateSpaceModel{T})
	check_dimensions(model)
	filt = kalman_filter(y, model)
	n = size(y, 1)
	x_filt = filt.filtered'
	x_pred = filt.predicted'
	x_smooth = zeros(size(x_filt))
	P = filt.error_cov
	P_smoov = zeros(size(P))
	model = filt.model

	P_pred = model.F * P[:, :, n-1] * model.F' + model.V
	J = P[:, :, n] * model.F' * inv(P_pred)
	x_smooth[:, n] = x_filt[:, n]
	for i = (n-1):-1:1
		J = P[:, :, i] * model.F' * inv(P_pred)
		x_smooth[:, i] = x_filt[:, i] + J * (x_smooth[:, i+1] - x_pred[:, i+1])
		P_smoov[:, :, i] = P[:, :, i] * J * (P_smoov[:, :, i+1] - P_pred) * J'
	end
	P_pred = model.F * model.P0 * model.F' + model.V
	J = P[:, :, 1] * model.F' * inv(P_pred)
	x_smooth[:, 1] = x_filt[:, 1] + J * (x_smooth[:, 2] - x_pred[:, 2])
	P_smoov[:, :, 1] = P[:, :, 1] * J * (P_smoov[:, :, 2] - P_pred) * J'

	return KalmanSmoothed(x_filt', x_pred', x_smooth, P_smoov[:,:,end],
		model, y, filt.loglik)
end