# Julia GARCH package
# Copyright 2013 Andrey Kolev
# Distributed under MIT license (see LICENSE.md)

module GARCH
using NLopt, Distributions

export garchFit, predict

type GarchFit
    data::Vector
    params::Vector
    llh::Float64
    status::Symbol
    converged::Bool
    sigma::Vector
    hessian::Array{Float64,2}
    cvar::Array{Float64,2}
    secoef::Vector
    tval::Vector
end

function Base.show(io::IO ,fit::GarchFit)
    pnorm(x) = 0.5 * (1 + erf(x / sqrt(2)))
    prt(x) = 2 * (1 - pnorm(abs(x)))
    jbstat, jbp = jbtest(fit.data./fit.sigma)

    @printf io "Fitted garch model \n"
    @printf io " * Coefficient(s):    %-15s%-15s%-15s\n" "α₀" "α₁" "β₁"
    @printf io "%-22s%-15.5g%-15.5g%-15.5g\n" "" fit.params[1] fit.params[2] fit.params[3]
    @printf io " * Log Likelihood: %.5g\n" fit.llh
    @printf io " * Converged: %s\n" fit.converged
    @printf io " * Solver status: %s\n\n" fit.status
    @printf io " * Standardised Residuals Tests:\n"
    @printf io "   %-26s%-15s%-15s\n" "" "Statistic" "tp-Value"
    @printf io "   %-21s%-5s%-15.5g%-15.5g\n\n" "Jarque-Bera Test" "χ²" jbstat jbp
    @printf io " * Error Analysis:\n"
    @printf io "   %-7s%-15s%-15s%-15s%-15s\n" "" "Estimate" "Std.Error" "t value" "Pr(>|t|)"
    @printf io "   %-7s%-15.5g%-15.5g%-15.5g%-15.5g\n" "α₀" fit.params[1] fit.secoef[1] fit.tval[1] prt(fit.tval[1])
    @printf io "   %-7s%-15.5g%-15.5g%-15.5g%-15.5g\n" "α₁" fit.params[2] fit.secoef[2] fit.tval[2] prt(fit.tval[2])
    @printf io "   %-7s%-15.5g%-15.5g%-15.5g%-15.5g\n" "β₁" fit.params[3] fit.secoef[3] fit.tval[3] prt(fit.tval[3])
end

function cdHessian(par, LLH)
    eps = 1e-4 * par
    n = length(par)
    H = zeros(n, n)
    for i in 1:n
        for j in 1:n
            x₁ = copy(par)
            x₁[i] += eps[i]
            x₁[j] += eps[j]
            x₂ = copy(par)
            x₂[i] += eps[i]
            x₂[j] -= eps[j]
            x₃ = copy(par)
            x₃[i] -= eps[i]
            x₃[j] += eps[j]
            x₄ = copy(par)
            x₄[i] -= eps[i]
            x₄[j] -= eps[j]
            H[i,j] = (LLH(x₁) - LLH(x₂) - LLH(x₃) + LLH(x₄)) / (4 .* eps[i] * eps[j])
        end
    end
    return H
end


function calculateVolatilityProcess(ɛ²::Vector, α₀, α₁, β₁)
    h = similar(ɛ²)
    h[1] = mean(ɛ²)
    for i = 2:length(ɛ²)
        h[i] = α₀ + α₁ * ɛ²[i - 1] + β₁ * h[i - 1]
    end
    return h
end


function garchLLH(y::Vector, x::Vector)
    ɛ² = y .^ 2
    T = length(y)
    α₀, α₁, β₁ = x
    h = calculateVolatilityProcess(ɛ², α₀, α₁, β₁)
    return -0.5 * (T - 1) * log(2π) - 0.5 * sum(log(h) + (y ./ sqrt(h)) .^ 2)
end

function predict(fit::GarchFit)
    α₀, α₁, β₁ = fit.params
    y = fit.data
    ɛ² = y.^2
    h = calculateVolatilityProcess(ɛ², α₀, α₁, β₁)
    return sqrt(α₀ + α₁ * ɛ²[end] + β₁ * h[end])
end

function garchFit(y::Vector)
    ɛ² = y.^2
    T = length(y)
    h = zeros(T)
    function garchLike(x::Vector, grad::Vector)
        α₀, α₁, β₁ = x
        h = calculateVolatilityProcess(ɛ², α₀, α₁, β₁)
        sum(log(h) + (y ./ sqrt(h)) .^ 2)
    end
    opt = Opt(:LN_SBPLX, 3)
    lower_bounds!(opt, [1e-10, 0.0, 0.0])
    upper_bounds!(opt, [1, 0.3, 0.99])
    min_objective!(opt, garchLike)
    (minf, minx, ret) = NLopt.optimize(opt, [1e-5, 0.09, 0.89])
    converged = minx[1] > 0 && all(minx[2:3] .>= 0) && sum(minx[2:3]) < 1.0
    H = cdHessian(minx, x -> garchLLH(y, x))
    cvar = -inv(H)
    secoef = sqrt(diag(cvar))
    tval = minx ./ secoef
    return GarchFit(y, minx, -0.5 * (T - 1) * log(2π) - 0.5 * minf, ret, converged, sqrt(h), H, cvar, secoef, tval)
end

end #GARCH
