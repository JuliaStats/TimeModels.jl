type StateSpaceModelMask
    # Process transition and noise covariance
    F::BitArray{2}
    A::BitArray{2}
    V::BitArray{2}
    # Observation and noise covariance
    G::BitArray{2}
    B::BitArray{2}
    W::BitArray{2}
    # Inital guesses at state and error covariance
    x1::BitArray{1}
    P1::BitArray{2}

    function StateSpaceModelMask(
            F::BitArray{2}, A::BitArray{2}, V::BitArray{2},
            G::BitArray{2}, B::BitArray{2}, W::BitArray{2}, 
            x1::BitArray{1}, P1::BitArray{2})

        nx = size(F, 1) 
        ny = size(G, 1)

        @assert size(F, 1) == nx 
        @assert size(F, 2) == nx 
        @assert size(A, 1) == nx 
        @assert size(V, 1) == nx 

        @assert size(G, 1) == ny
        @assert size(G, 2) == nx 
        @assert size(B, 1) == ny
        @assert size(W, 1) == ny

        @assert length(x1) == nx
        @assert size(P1, 1) == nx 

        @assert size(A, 2) == size(B, 2)

        new(F, A, V, G, B, W, x1, P1)
    end
end

function StateSpaceModelMask(
        F::BitArray{2}, V::BitArray{2},
        G::BitArray{2}, W::BitArray{2},
        x1::BitArray{1}, P1::BitArray{2})
	  StateSpaceModelMask(F, falses(size(F,1),1), V, G, falses(size(G,1),1), W, x1, P1)
end

type StateSpaceModel{T}
    # Process transition and noise covariance
    F::Union(Matrix{T}, Matrix{Function})
    A::Matrix{T}
    V::Matrix{T}
    # Observation and noise covariance
    G::Union(Matrix{T}, Matrix{Function})
    B::Matrix{T}
    W::Matrix{T}
    # Inital guesses at state and error covariance
    x1::Vector{T}
    P1::Matrix{T}

    fitmask::StateSpaceModelMask

    function StateSpaceModel(
            F::Union(Matrix{T}, Matrix{Function}), A::Matrix{T}, V::Matrix{T},
            G::Union(Matrix{T}, Matrix{Function}), B::Matrix{T}, W::Matrix{T}, 
            x1::Vector{T}, P1::Matrix{T}, mask::StateSpaceModelMask)

        ispossemidef(x::Matrix) = issym(x) && (eigmin(x) >= 0)
        nx = size(F, 1) 
        ny = size(G, 1)

        @assert size(F, 1) == nx 
        @assert size(F, 2) == nx 
        @assert size(A, 1) == nx 
        @assert size(V, 1) == nx 
        @assert ispossemidef(V)

        @assert size(G, 1) == ny
        @assert size(G, 2) == nx 
        @assert size(B, 1) == ny
        @assert size(W, 1) == ny
        @assert ispossemidef(W)

        @assert length(x1) == nx
        @assert size(P1, 1) == nx 
        @assert ispossemidef(P1)

        @assert size(A, 2) == size(B, 2)

        @assert (size(F), size(A), size(V)) == (size(mask.F), size(mask.A), size(mask.V))
        @assert (size(G), size(B), size(W)) == (size(mask.G), size(mask.B), size(mask.W))
        @assert (size(x1), size(P1)) == (size(mask.x1), size(mask.P1))

        new(F, A, V, G, B, W, x1, P1, mask)
    end
end

function StateSpaceModel{T<:Real}(
        F::Union(Matrix{T}, Matrix{Function}), V::Matrix{T},
        G::Union(Matrix{T}, Matrix{Function}), W::Matrix{T},
        x1::Vector{T}, P1::Matrix{T})

    mask = StateSpaceModelMask(trues(F), eye(V) .> 0, trues(G), eye(W) .> 0, trues(x1), trues(P1))
	  StateSpaceModel{T}(F, zeros(size(F,1),1), V, G, zeros(size(G,1),1), W, x1, P1, mask)
end

function show{T}(io::IO, mod::StateSpaceModel{T})
    dx, dy = length(mod.x1), size(mod.G, 1)
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
    pred_error_cov::Array{T}
    kalman_gain::Array{T}
    model::StateSpaceModel
    y::Array{T}
    loglik::T
end

function show{T}(io::IO, filt::KalmanFiltered{T})
    n = size(filt.y, 1)
    dx, dy = length(filt.model.x1), size(filt.model.G, 1)
    println("KalmanFiltered{$T}")
    println("$n observations, $dx-D process x $dy-D observations")
    println("Negative log-likelihood: $(filt.loglik)")
end

type KalmanSmoothed{T}
    predicted::Array{T}
    filtered::Array{T}
    smoothed::Array{T}
    error_cov::Array{T}
    error_cov_lag1::Array{T}
    model::StateSpaceModel
    y::Array{T}
    loglik::T
end

function show{T}(io::IO, filt::KalmanSmoothed{T})
    n = size(filt.y, 1)
    dx, dy = length(filt.model.x1), size(filt.model.G, 1)
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

function simulate{T}(model::StateSpaceModel{T}, n::Int; u::Array{T}=zeros(n, size(model.A, 2)))
    # Generates a realization of a state space model.
    #
    # Arguments:
    # model : StateSpaceModel
    #	Model defining the process
    # n : Int
    #	Number of steps to simulate.
    # u : Matrix
    # Optional control input sequence

    u = u'
    @assert size(u, 1) == size(model.A, 2)
    @assert size(u, 1) == size(model.B, 2)
    @assert size(u, 2) == n

    # dimensions of the process and observation series
    nx = length(model.x1)
    ny = size(model.G, 1)

    # create empty arrays to hold the state and observed series
    x = zeros(nx, n)
    y = zeros(ny, n)

    # Cholesky decompositions of the covariance matrices, for generating
    # random noise
    V_chol = @compat chol(model.V, Val{:L})
    W_chol = @compat chol(model.W, Val{:L})

    # Generate the series
    x[:, 1] = model.x1
    y[:, 1] = eval_matrix(model.G, 1, T)*x[:, 1] + model.B*u[:,1] + W_chol*randn(ny)
    for i=2:n
        x[:, i] = eval_matrix(model.F, i-1, T)*x[:, i-1] + model.A*u[:,i-1] + V_chol*randn(nx)
        y[:, i] = eval_matrix(model.G, i, T)*x[:, i] + model.B*u[:,i] + W_chol*randn(ny)
    end

    return x', y'
end

function kalman_filter{T}(y::Array{T}, model::StateSpaceModel{T};
          u::Array{T}=zeros(size(y,1), size(model.A, 2)))

	  function kalman_recursions(y_i, u_i, G_i, B, W, x_pred_i, P_pred_i)
        if !any(isnan(y_i))
            innov =  y_i - G_i * x_pred_i - B * u_i
            S = G_i * P_pred_i * G_i' + W  # Innovation covariance
            Kgain = P_pred_i * G_i' / S # Kalman gain
            x_filt_i = x_pred_i + Kgain * innov
            P_filt_i = (I - Kgain * G_i) * P_pred_i
            dll = (dot(innov,S\innov) + logdet(S))/2
        else
            Kgain = zeros(length(x_pred_i), length(y_i))
            x_filt_i = x_pred_i
            P_filt_i = P_pred_i
            dll = 0
        end
        return x_filt_i, P_filt_i, Kgain[:,:], dll
    end #kalman_recursions

    y = y'
    u = u'

    n = size(y, 2)
    nx = length(model.x1)
    ny = size(y, 1)
    nu = size(u, 1)

    @assert size(model.G, 1) == ny
    @assert size(model.A, 2) == nu
    @assert size(model.B, 2) == nu
    @assert size(u, 2) == n

    x_pred = zeros(nx, n)
    x_filt = zeros(x_pred)
    P_pred = zeros(size(model.P1, 1), size(model.P1, 2), n)
    P_filt = zeros(P_pred)
    K = zeros(nx, ny, n)
    log_likelihood = n*ny*log(2pi)/2

    # first iteration
    F_1 = eval_matrix(model.F, 1, T)
    x_pred[:, 1] = model.x1
    P_pred[:, :, 1] = model.P1
    x_filt[:, 1], P_filt[:,:,1], K[:,:,1], dll = kalman_recursions(y[:, 1], u[:,1], 
        eval_matrix(model.G, 1, T), model.B, model.W, x_pred[:,1], P_pred[:,:,1])
    log_likelihood += dll
    for i=2:n
        F_i = eval_matrix(model.F, i, T)
        x_pred[:, i] =  F_i * x_filt[:, i-1] + model.A * u[:,i-1]
        P_pred[:, :, i] = F_i * P_filt[:, :, i-1] * F_i' + model.V
        x_filt[:, i], P_filt[:,:,i], K[:,:,i], dll = kalman_recursions(y[:, i], u[:,i],
            eval_matrix(model.G, i, T), model.B, model.W, x_pred[:,i], P_pred[:,:,i])
        log_likelihood += dll
    end

    return KalmanFiltered(x_filt', x_pred', P_filt, P_pred, K, model, y', log_likelihood)
end


function kalman_smooth{T}(y::Array{T}, model::StateSpaceModel{T};
          u::Array{T}=zeros(size(y,1), size(model.A, 2)))
    filt = kalman_filter(y, model, u=u)
    n = size(y, 1)
    x_pred = filt.predicted'
    x_filt = filt.filtered'
    x_smooth = zeros(size(x_filt))
    P_pred = filt.pred_error_cov
    P_filt = filt.error_cov
    P_smoov = zeros(P_filt)
    P_tt1 = zeros(P_filt)
    J = zeros(P_filt)

    x_smooth[:, n] = x_filt[:, n]
    P_smoov[:, :, n] = P_filt[:, :, n]

    for i = (n-1):-1:1
        F_i = eval_matrix(model.F, i, T)
        J[:,:,i+1] = P_filt[:, :, i] * F_i' * inv(P_pred[:,:,i+1])
        x_smooth[:, i] = x_filt[:, i] + J[:,:,i+1] * (x_smooth[:, i+1] - x_pred[:, i+1])
        P_smoov[:, :, i] = P_filt[:, :, i] + 
            J[:,:,i+1] * (P_smoov[:, :, i+1] - P_pred[:,:,i+1]) * J[:,:,i+1]'
    end

    J[:,:,1] = model.P1 * eval_matrix(model.F, 1, T)' * inv(model.P1)
    P_tt1[:, :, n] = (I - filt.kalman_gain[:, :, n] * eval_matrix(model.G, n, T)) * 
                        eval_matrix(model.F, n, T) * P_filt[:, :, n-1]
 
    for i = (n-1):-1:1
        P_tt1[:, :, i] = P_filt[:, :, i] * J[:, :, i]' + 
                            J[:, :, i+1] * (P_tt1[:, :, i+1] - eval_matrix(model.F, n, T) * 
                            P_filt[:, :, i]) * J[:, :, i]
    end #for

    return KalmanSmoothed(x_pred', x_filt', x_smooth', P_smoov, P_tt1, model, y, filt.loglik)
end

# ML parameter estimation with filter result log-likelihood (via Nelder-Mead)
function fit{T}(y::Matrix{T}, build::Function, theta0::Vector{T};
          u::Array{T}=zeros(size(y,1), size(build(theta0).A, 2)))
    objective(theta) = kalman_smooth(y, build(theta), u=u).loglik
    kfit = Optim.optimize(objective, theta0)
    return (kfit.minimum, build(kfit.minimum), kalman_smooth(y, build(kfit.minimum)))
end #fit

# Expectation-Maximization (EM) parameter estimation
function fit{T}(y::Matrix{T}, model::StateSpaceModel;
          u::Array{T}=zeros(size(y,1), size(model.A, 2)), eps::Float64=1e-6, niter::Int=typemax(Int))

    y = y'
    u = u'
    n = size(y,2)
    has_input = any((x->x!=0),u)

    ut_ut_tail  = u[:, 2:end] * u[:, 2:end]'
    ut_ut       = ut_ut_tail + u[:, 1] * u[:, 1]'
    yt_yt       = y * y'
    yt_ut       = y * u'

    function maximize(expectations::KalmanSmoothed)

        x           = expectations.smoothed'
        xt_tail     = x[:, 2:end] * x[:, 2:end]'
        xt          = xt_tail + x[:,1] * x[:,1]' 
        xt1         = x[1:end-1] * x[1:end-1]'
        xtt1        = x[2:end] * x[1:end-1]'
        Pt_tail     = reshape(sum(expectations.error_cov[:,:,2:end], 3), 
                        size(expectations.error_cov)[1:2])
        Pt          = Pt_tail + expectations.error_cov[:,:,1]
        invPt       = inv(Pt)
        Pt1         = reshape(sum(expectations.error_cov[:,:,1:end-1], 3),
                        size(expectations.error_cov)[1:2])
        invPt1      = inv(Pt1)
        Ptt1        = reshape(sum(expectations.error_cov_lag1[:,:,2:end], 3),
                        size(expectations.error_cov_lag1)[1:2])
        ut_xt_tail  = u[:, 2:end] * x[:, 2:end]'
        ut_xt       = ut_xt_tail + u[:, 1] * x[:, 1]'
        ut_xt1      = u[:, 2:end] * x[:, 1:end-1]'
        yt_xt       = y * x' 

        # TODO: A and B estimations are based on the unfixed parameter estimates for F and G.
        # Need to account for fact that some parameters might be fixed to other values.
        # As such, A and B estimates (when not fixed) are not yet likely to be reliable.
        A = has_input ? 
            (ut_xt_tail' - (xtt1 + Ptt1) * inv(xt1 + Pt1) * ut_xt1') * inv(ut_ut_tail - ut_xt1 * inv(xt1 + Pt1) * ut_xt1') : 
            expectations.model.A
        A = expectations.model.fitmask.A .* A + !expectations.model.fitmask.A .* expectations.model.A

        F = (x[:, 2:end] * x[:, 1:end-1]' + Ptt1 - A * ut_xt1) * inv(x[:, 1:end-1] * x[:, 1:end-1]' + Pt1)
        F = expectations.model.fitmask.F .* F + !expectations.model.fitmask.F .* expectations.model.F

        x_innov = x[:, 2:end] - F * x[:, 1:end-1] - A * u[:, 2:end]
        V = 1/(n+1) * (x_innov * x_innov' + Pt_tail + F * Pt1 * F' + F * Ptt1' + Ptt1 * F')
        V = expectations.model.fitmask.V .* V + !expectations.model.fitmask.V .* expectations.model.V

        B = has_input ? 
            (yt_ut - yt_xt * inv(xt + Pt) * ut_xt') * inv(ut_ut - ut_xt * inv(xt + Pt) * ut_xt') :
            expectations.model.B
        B = expectations.model.fitmask.B .* B + !expectations.model.fitmask.B .* expectations.model.B

        G = (yt_xt - B * ut_xt) * inv(x * x' + Pt)
        G = expectations.model.fitmask.G .* G + !expectations.model.fitmask.G .* expectations.model.G

        y_innov = y - G * x - B * u
        W = 1/n * (y_innov * y_innov' + G * Pt * G')
        W = expectations.model.fitmask.W .* W + !expectations.model.fitmask.W .* expectations.model.W

        x1 = x[:,1]
        x1 = expectations.model.fitmask.x1 .* x1 + !expectations.model.fitmask.x1 .* expectations.model.x1

        P1 = expectations.error_cov[:,:,1]
        P1 = expectations.model.fitmask.P1 .* P1 + !expectations.model.fitmask.P1 .* expectations.model.P1

        return StateSpaceModel{T}(F,A,V,G,B,W,x1,P1, expectations.model.fitmask)

    end #maximize

    fraction_change(ll_prev, ll_curr) = isinf(ll_prev) ? 1 : 2 * (ll_prev - ll_curr) / (ll_prev + ll_curr + 1e-6) 

    ll_prev = Inf 
    expectations = kalman_smooth(y', model, u=u')
    println(expectations.loglik)
    model = maximize(expectations)
    niter -= 1

    while (niter > 0) & (fraction_change(ll_prev, expectations.loglik) > eps)
        ll_prev = expectations.loglik
        expectations = kalman_smooth(y', model, u=u')
        model = maximize(expectations)
        niter -= 1
    end #while

    return model, kalman_smooth(y', model, u=u')
end #fit
