function fit{T}(y::Matrix{T}, build::Function, theta0::Vector{T})
    objective(theta) = kalman_filter(y, build(theta)).loglik
    kfit = Optim.optimize(objective, theta0)
    return (kfit.minimum, build(kfit.minimum))
end
