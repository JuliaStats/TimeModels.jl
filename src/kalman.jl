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

function simulate_statespace(n, model, x0)
	# Generates a realization of a state space model.
	#
	# Arguments:
	# n : Int
	#	Number of steps to simulate.
	# model : StateSpaceModel
	#	Model defining the process
	# x0 : Array{float}.
	#	Inital state vector.
	return None
end

function KalmanFilter(y, model)
	n = size(x, 1)
	x_est = zeros(length(model.x0), n)
	x_est[1, :] = model.x0
	P0 = model.P0
	for i=1:n
		
	end
end