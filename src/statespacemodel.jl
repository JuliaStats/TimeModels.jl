immutable StateSpaceModel{T <: Real}

    # Process transition matrix, control matrix, and noise covariance
    F::Function
    B::Matrix{T}
    V::Matrix{T}

    # Observation matrix, feed-forward matrix, and noise covariance
    G::Function
    D::Matrix{T}
    W::Matrix{T}

    # Inital state and error covariance conditions
    x1::Vector{T}
    P1::Matrix{T}

    # Model dimensions
    nx::Int
    ny::Int
    nu::Int

    function StateSpaceModel(F::Function, B::Matrix{T}, V::Matrix{T},
                    G::Function, D::Matrix{T}, W::Matrix{T}, x1::Vector{T}, P1::Matrix{T})

        ispossemidef(x::Matrix) = issym(x) && (eigmin(x) >= 0)
        @assert ispossemidef(V)
        @assert ispossemidef(W)
        @assert ispossemidef(P1)

        nx, ny, nu = confirm_matrix_sizes(F(1), B, V, G(1), D, W, x1, P1)
        new(F, B, V, G, D, W, x1, P1, nx, ny, nu)
    end
end

# Time-dependent definitions
function StateSpaceModel{T}(F::Function, B::Matrix{T}, V::Matrix{T},
                            G::Function, D::Matrix{T}, W::Matrix{T},
                            x1::Vector{T}, P1::Matrix{T})
	  StateSpaceModel{T}(F, B, V, G, D, W, x1, P1)
end

function StateSpaceModel{T}(F::Function, V::Matrix{T},
                            G::Function, W::Matrix{T},
                            x1::Vector{T}, P1::Matrix{T})
    B = zeros(size(V, 1), 1)
    D = zeros(size(W, 1), 1)
    StateSpaceModel{T}(F, B, V, G, D, W, x1, P1)
end

# Time-independent definitions
function StateSpaceModel{T}(F::Matrix{T}, B::Matrix{T}, V::Matrix{T},
                            G::Matrix{T}, D::Matrix{T}, W::Matrix{T},
                            x1::Vector{T}, P1::Matrix{T})
	StateSpaceModel{T}(_->F, B, V, _->G, D, W, x1, P1)
end

function StateSpaceModel{T}(F::Matrix{T}, V::Matrix{T},
                            G::Matrix{T}, W::Matrix{T},
                            x1::Vector{T}, P1::Matrix{T})
    B = zeros(size(V, 1), 1)
    D = zeros(size(W, 1), 1)
    StateSpaceModel{T}(_->F, B, V, _->G, D, W, x1, P1)
end

function show{T}(io::IO, mod::StateSpaceModel{T})
    dx, dy = length(mod.x1), size(mod.G, 1)
    println("StateSpaceModel{$T}, $dx-D process x $dy-D observations")
    println("Process evolution matrix F:")
    show(mod.F)
    if any(mod.B .!= 0)
        println("\n\nControl input matrix B:")
        show(mod.B)
    end #if
    println("\n\nProcess error covariance V:")
    show(mod.V)
    println("\n\nObservation matrix G:")
    show(mod.G)
    if any(mod.D .!= 0)
        println("\n\nFeed-forward matrix D:")
        show(mod.D)
    end #if
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
    V_chol = chol(model.V, Val{:L})
    W_chol = chol(model.W, Val{:L})

    # Generate the series
    x[:, 1] = model.x1
    y[:, 1] = model.G(1) * model.x1 + model.D * u[:, 1] + W_chol * randn(model.ny)
    for i=2:n
        x[:, i] = model.F(i-1) * x[:, i-1] + model.B * u[:, i-1] + V_chol * randn(model.nx)
        y[:, i] = model.G(i)   * x[:, i]   + model.D * u[:, i]   + W_chol * randn(model.ny)
    end

    return x', y'
end

