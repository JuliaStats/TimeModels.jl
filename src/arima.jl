module arima

export arima_statespace, arima

using Kalman

# Constructs a state-space representation of an ARIMA model given
# vectors of AR and MA coefficients and a number of differences.
# This implementation is based on Section 8.3 in Brockwell and Davis 2002,
# "Introduction to Time Series and Forecasting," 2nd ed.
function arima_statespace{T, I<:Integer}(ar::Vector{T}, d::I, ma::Vector{T}, 
		sigma::T)
	p = length(ar)
	q = length(ma)
	r = max(p, q + 1)

	F = [zeros(r - 1) diagm(ones(r - 1)); [zeros(max(0, r - p)), reverse(ar)]']
	V = diagm([zeros(r - 1), sigma])
	G = [zeros(max(0, r - q - 1)), reverse(ma), 1]'
	W = zeros(1, 1)
	# differencing
	if d > 0
		A = [zeros(d - 1), 1]
		B = [zeros(d - 1) diagm(ones(d - 1)); 
			 T[(-1.0)^(d + 1 - i) * binomial(d, d-i) for i in 0:(d - 1)]']
		# redefine matrices to incorporate differencing into state
		F = [F zeros(r, d); A*G B]
		V = [V zeros(r, d); zeros(d, r + d)]
		G = [G B[end, :]]
		x0 = zeros(r + d)
		P0 = diagm(ones(r + d) * 1e7)
	else
		x0 = zeros()
		P0 = diagm(ones(r) * 1e7)
	end
	StateSpaceModel(F, V, G, W, x0, P0)
end

function arima{T, I<:Integer}(y::Vector{T}, p::I, d::I, q::I)
	build(par::Vector{T}) = arima_statespace(par[1:p], par[p+1], par[p+2:end])
	par0 = zeros(p + d + q)
	fit(y, build, par0)
end


end # module