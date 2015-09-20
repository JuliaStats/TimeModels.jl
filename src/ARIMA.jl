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
	fit(y'', build, par0)
end

function ar{T}(y::Vector{T}, p::Int; niter::Int=1000, eps::Float64=1e-5)

    @assert p > 0

    y = y .- mean(y)

    lse = y[p+1:end]
    for t in 1:p
        lse = [lse y[p-t+1:end-t]]
    end #for

    lse = lse[find(!any(isnan(lse), 2)), :]
    phi_start = llsq(lse[:,2:end], lse[:,1], bias=false)

    B_f = [zeros(p)'; eye(p-1) zeros(p-1)][:]
    B_D = zeros(p^2, p)
    B_D[[1 + (i-1)*(p^2 + p) for i in 1:p]] = 1
    B = ParametrizedMatrix(B_f, B_D, (p,p))

    pmatrices, params = zip(
        (B, phi_start),
        parametrize_none(1e-9*eye(p)),
        parametrize_none([[1.] zeros(p-1)']),
        parametrize_none([10.]'),
        parametrize_none(y[1:p]),
        parametrize_none(1e-9*eye(p))
    )

    params, _ = fit(y[(p+1):end], ParametrizedSSM(pmatrices...), SSMParameters(params...), niter=niter, eps=eps)
    return params.B

end #ar

function arx{T}(y::Vector{T}, u::Array{T}, p::Int; niter::Int=1000, eps::Float64=1e-5)

    nu = size(u, 2)
    @assert p > 0
    @assert nu > 0

    y = y .- mean(y)

    lse = y[p+1:end]
    for t in 1:p
        lse = [lse y[p-t+1:end-t]]
    end #for

    lse = [lse u[p+1:end, :]]
    lse = lse[find(!any(isnan(lse), 2)), :]
    llsq_fit = llsq(lse[:,2:end], lse[:,1], bias=false)
    phi_start, beta_start = llsq_fit[1:p], llsq_fit[p+1:end]

    B_f = [zeros(p)'; eye(p-1) zeros(p-1)][:]
    B_D = zeros(p^2, p)
    B_D[[1 + (i-1)*(p^2 + p) for i in 1:p]] = 1
    B = ParametrizedMatrix(B_f, B_D, (p,p))

    pmatrices, params = zip(
        (B, phi_start),
        parametrize_none(zeros(p, nu)),
        parametrize_none(1e-5*eye(p)),
        parametrize_none([[1.] zeros(p-1)']),
        parametrize_full(beta_start'),
        parametrize_none([10.]'),
        parametrize_none(y[1:p]),
        parametrize_none(1e-5*eye(p))
    )

    params, _ = fit(y[(p+1):end], ParametrizedSSM(pmatrices...), SSMParameters(params...), u=u[(p+1):end,:], niter=niter, eps=eps)
    return params.B, params.A

end #arx
