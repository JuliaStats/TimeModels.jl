immutable StateSpaceModel{T <: Real}

    # Process transition matrix, control matrix, and noise covariance
    F::Function
    B::Function
    V::Function

    # Observation matrix, feed-forward matrix, and noise covariance
    G::Function
    D::Function
    W::Function

    # Inital state and error covariance conditions
    x1::Vector{T}
    P1::Matrix{T}

    # Model dimensions
    nx::Int
    ny::Int
    nu::Int

    function StateSpaceModel(F::Function, B::Function, V::Function,
                    G::Function, D::Function, W::Function, x1::Vector{T}, P1::Matrix{T})

        ispossemidef(x::Matrix) = issym(x) && (eigmin(x) >= 0)
        @assert ispossemidef(V(1))
        @assert ispossemidef(W(1))
        @assert ispossemidef(P1)

        nx, ny, nu = confirm_matrix_sizes(F(1), B(1), V(1), G(1), D(1), W(1), x1, P1)
        new(F, B, V, G, D, W, x1, P1, nx, ny, nu)
    end
end

# Time-dependent definitions
function StateSpaceModel{T}(F::Function, B::Function, V::Function,
                            G::Function, D::Function, W::Function,
                            x1::Vector{T}, P1::Matrix{T})
	  StateSpaceModel{T}(F, B, V, G, D, W, x1, P1)
end

function StateSpaceModel{T}(F::Function, V::Function,
                            G::Function, W::Function,
                            x1::Vector{T}, P1::Matrix{T})
    B(_) = zeros(size(V(1), 1), 1)
    D(_) = zeros(size(W(1), 1), 1)
    StateSpaceModel{T}(F, B, V, G, D, W, x1, P1)
end

# Time-independent definitions
function StateSpaceModel{T}(F::Matrix{T}, B::Matrix{T}, V::Matrix{T},
                            G::Matrix{T}, D::Matrix{T}, W::Matrix{T},
                            x1::Vector{T}, P1::Matrix{T})
	StateSpaceModel{T}(_->F, _->B, _->V, _->G, _->D, _->W, x1, P1)
end

function StateSpaceModel{T}(F::Matrix{T}, V::Matrix{T},
                            G::Matrix{T}, W::Matrix{T},
                            x1::Vector{T}, P1::Matrix{T})
    B(_) = zeros(size(V, 1), 1)
    D(_) = zeros(size(W, 1), 1)
    StateSpaceModel{T}(_->F, B, _->V, _->G, D, _->W, x1, P1)
end

function show{T}(io::IO, mod::StateSpaceModel{T})
    dx, dy = length(mod.x1), size(mod.G, 1)
    println("StateSpaceModel{$T}, $dx-D process x $dy-D observations")
    println("Process evolution matrix F:")
    show(mod.F)
    println("\n\nControl input matrix B:")
    show(mod.B)
    println("\n\nProcess error covariance V:")
    show(mod.V)
    println("\n\nObservation matrix G:")
    show(mod.G)
    println("\n\nFeed-forward matrix D:")
    show(mod.D)
    println("\n\nObseration error covariance W:")
    show(mod.W)
end

function confirm_matrix_sizes(F, B, V, G, D, W, x1, P1)

    nx = size(F, 1)
    nu = size(B, 2)
    ny = size(G, 1)

    @assert size(F) == (nx, nx)
    @assert size(B) == (nx, nu)
    @assert size(V) == (nx, nx)

    @assert size(G) == (ny, nx)
    @assert size(D) == (ny, nu)
    @assert size(W) == (ny, ny)

    @assert length(x1) == nx
    @assert size(P1) == (nx, nx)

    return nx, ny, nu

end #confirm_matrix_sizes

function simulate(model::StateSpaceModel, n::Int; u::Array=zeros(n, model.nu))
    # Generates a realization of a state space model.
    #
    # Arguments:
    # model : StateSpaceModel
    #	Model defining the process
    # n : Int
    #	Number of steps to simulate.
    # u : Array (optional)
    # n x nu array of exogenous inputs

    @assert size(u, 1) == n
    @assert size(u, 2) == model.nu

    # create empty arrays to hold the state and observed series
    x = zeros(model.nx, n)
    y = zeros(model.ny, n)
    u = u'

    # Cholesky decompositions of the covariance matrices, for generating
    # random noise
    V_chol(t) = chol(model.V(t), Val{:L})
    W_chol(t) = chol(model.W(t), Val{:L})

    # Generate the series
    x[:, 1] = model.x1
    y[:, 1] = model.G(1) * model.x1 + model.D(1) * u[:, 1] + W_chol(1) * randn(model.ny)
    for i=2:n
        x[:, i] = model.F(i-1) * x[:, i-1] + model.B(i-1) * u[:, i-1] + V_chol(i-1) * randn(model.nx)
        y[:, i] = model.G(i)   * x[:, i]   + model.D(i)   * u[:, i]   + W_chol(1) * randn(model.ny)
    end

    return x', y'
end

