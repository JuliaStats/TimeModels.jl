using Optim
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

function issquare(x::Array)
	return size(x, 1) == size(x, 2)
end

function simulate_statespace(n, model)
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

function check_dimensions(model)
	@assert length(model.x0) == size(model.F, 2)
	@assert size(model.F) == size(model.V)
	@assert issquare(model.F)
	@assert size(model.G, 2) == size(model.F, 1)
	@assert size(model.G, 1) == size(model.W, 1)
	@assert issquare(model.W)
	@assert size(model.P0, 1) == length(model.x0)
	@assert issquare(model.P0)
end

function KalmanFilter(y, model)
	check_dimensions(model)
	n = size(y, 1)
	y = y'
	x_est = zeros(length(model.x0), n)
	x_est[:, 1] = model.F * model.x0
	P0 = model.P0
	I = eye(size(x_est, 1))
	for i=2:n
		# prediction
		x_est[:, i] = model.F * x_est[:, i-1]
		P1 = model.F * P0 * model.F' + model.V

		innovation =  y[:, i] - model.G * x_est[:, i]
		S = model.G * P1 * model.G' + model.W 	# Innovation covariance
		K = P1 * model.G' * inv(S)				# Kalman gain
		x_est[:, i] = x_est[:, i] + K * innovation
		P0 = (I - K * model.G) * P1
	end
	return x_est'
end
