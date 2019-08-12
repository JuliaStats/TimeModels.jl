abstract type AbstractStateSpaceModel{T<:Real} end

struct StateSpaceModel{T} <: AbstractStateSpaceModel{T}

    # Process transition matrix, control matrix, and noise covariance
    A::Function
    B::Function
    V::Function

    # Observation matrix, feed-forward matrix, and noise covariance
    C::Function
    D::Function
    W::Function

    # Inital state and error covariance conditions
    x1::Vector{T}
    P1::Union{Matrix{T}, SparseMatrixCSC{T}}

    # Model dimensions
    nx::Int
    ny::Int
    nu::Int

    function StateSpaceModel{T}(A::Function, B::Function, V::Function,
                                C::Function, D::Function, W::Function,
                                x1::Vector{T}, P1::Union{Matrix{T}, SparseMatrixCSC{T}}) where {T}

        ispossemidef(x::AbstractMatrix) = issymmetric(x) && (eigmin(Matrix(x)) >= 0)
        @assert ispossemidef(V(1))
        @assert ispossemidef(W(1))
        @assert ispossemidef(P1)

        nx, ny, nu = confirm_matrix_sizes(A(1), B(1), V(1), C(1), D(1), W(1), x1, P1)
        new(A, B, V, C, D, W, x1, P1, nx, ny, nu)
    end
end

# Time-dependent constructor
StateSpaceModel(A::Function, V::Function, C::Function, W::Function,
                       x1::Vector{T}, P1::Union{Matrix{T}, SparseMatrixCSC{T}};
                       B::Function=_->zeros(size(V(1), 1), 1),
                       D::Function=_->zeros(size(W(1), 1), 1)) where {T} =
	  StateSpaceModel{T}(A, B, V, C, D, W, x1, P1)

# Time-independent constructor
StateSpaceModel(A::Matrix{T}, V::Matrix{T}, C::Matrix{T}, W::Matrix{T},
                       x1::Vector{T}, P1::Union{Matrix{T}, SparseMatrixCSC{T}};
                       B::Matrix{T}=zeros(size(A, 1), 1),
                       D::Matrix{T}=zeros(size(C, 1), 1)) where {T} =
	  StateSpaceModel{T}(_->A, _->B, _->V, _->C, _->D, _->W, x1, P1)

function show(io::IO, mod::StateSpaceModel{T}) where T
    dx, dy = mod.nx, mod.ny
    println("StateSpaceModel{$T}, $dx-D process x $dy-D observations")
    println("Process evolution matrix A:")
    show(mod.A(1))
    println("\n\nControl input matrix B:")
    show(mod.B(1))
    println("\n\nProcess error covariance V:")
    show(mod.V(1))
    println("\n\nObservation matrix C:")
    show(mod.C(1))
    println("\n\nFeed-forward matrix D:")
    show(mod.D(1))
    println("\n\nObseration error covariance W:")
    show(mod.W(1))
end

# Time-independent parametrized matrix type
# TODO: A more general ParametrizedArray (allowing vectors) could be preferable
struct ParametrizedMatrix{T}

    f::AbstractVector{T}
    D::Union{Matrix{T}, SparseMatrixCSC{T}}
    np::Int
    dims::Tuple{Int, Int}

    function ParametrizedMatrix{T}(f::AbstractVector{T},
                D::Union{Matrix{T}, SparseMatrixCSC{T, Int}}, dims::Tuple{Int, Int}) where {T}
        @assert length(f) == size(D, 1)
        @assert length(f) == dims[1] * dims[2]
        new(f, D, size(D, 2), dims)
    end #ParametrizedMatrix

end #ParametrizedMatrix

ParametrizedMatrix(f::Vector{T}, D::Matrix{T}, dims::Tuple{Int, Int}) where {T} = ParametrizedMatrix{T}(f, D, dims)

function ParametrizedMatrix(f::SparseVector{T}, D::SparseMatrixCSC{T}, dims::Tuple{Int, Int}) where T
    ParametrizedMatrix{T}(f, D, dims)
end #ParametrizedMatrix

function show(io::IO, cpm::ParametrizedMatrix{T}) where T

    function combinestrelems(a, b)
        if (a != "") & (b != "")
          a * " + " * b
        elseif (a == "") & (b == "")
          "0"
        else
          a * b
        end #if
    end #combinestrelems

    nx, ny = cpm.dims
    println(nx, "x", ny, " ParametrizedMatrix{$T}")
    conststring = map(x -> x == 0 ? "" : string(x), cpm.f)
    paramstring = fill("", length(cpm.f))
    paramelems = fill("", cpm.np)

    for i = 1:size(cpm.D,1)
        for j = 1:cpm.np
            if  cpm.D[i,j] == 0
                paramelems[j] = ""
            elseif cpm.D[i,j] == 1
                paramelems[j] = string(Char(96+j))
            else
                paramelems[j] = string(cpm.D[i,j],Char(96+j))
            end #if
        end #for
        paramstring[i] = isempty(paramelems) ? "0" : reduce(combinestrelems, paramelems)
    end #for

    finalstrings = map(combinestrelems, conststring, paramstring)

    reshape(finalstrings, cpm.dims) |> showcompact
end

Base.length(pm::ParametrizedMatrix) = pm.dims[1] * pm.dims[2]
Base.size(pm::ParametrizedMatrix) = pm.dims
Base.size(pm::ParametrizedMatrix, dim::Int) = pm.dims[dim]

function (pm::ParametrizedMatrix{T})(params::Vector{T}) where {T}
    @assert pm.np == length(params)
    return pm.np == 0 ?
        reshape(pm.f, pm.dims) :
        reshape(pm.f + pm.D * sparse(params), pm.dims)
end #call

parametrize_full(X::Matrix{T}) where {T} =
        ParametrizedMatrix{T}(zeros(length(X)), Matrix(1.0I, length(X), length(X)), size(X))

function parametrize_diag(x::Vector{T}; sparse=false) where T
    n = length(x)
    f = sparse ? spzeros(n^2, 1) : zeros(n^2)
    D = sparse ? spzeros(n^2, n) : zeros(n^2, n)
    D[[1 + (i-1)*(n^2 + n + 1) for i in 1:n]] .= 1
    return ParametrizedMatrix{T}(f, D, (n,n))
end #parametrize_diag

parametrize_none(X::Matrix{T}) where {T} =
        ParametrizedMatrix{T}(vec(X), zeros(length(X), 0), size(X))

parametrize_none(X::SparseMatrixCSC{T}) where {T} =
        ParametrizedMatrix{T}(vec(X), spzeros(length(X), 0), size(X))

# Time-independent parametrized state space model
struct ParametrizedSSM{T} <: AbstractStateSpaceModel{T}

    # Transition equation and noise covariance

    A1::Function
    A2::ParametrizedMatrix{T}
    A3::Function

    B1::Function
    B2::ParametrizedMatrix{T}

    G::Function
    Q::ParametrizedMatrix{T}

    # Observation equation and noise covariance

    C1::Function
    C2::ParametrizedMatrix{T}
    C3::Function

    D1::Function
    D2::ParametrizedMatrix{T}

    H::Function
    R::ParametrizedMatrix{T}

    # Initial state and error covariance
    x1::ParametrizedMatrix{T}
    J::AbstractMatrix
    S::ParametrizedMatrix{T}

    nx::Int
    ny::Int
    nu::Int
    nq::Int
    nr::Int
    ns::Int

    function ParametrizedSSM{T}(A1::Function, A2::ParametrizedMatrix{T}, A3::Function,
                                B1::Function, B2::ParametrizedMatrix{T},
                                G::Function, Q::ParametrizedMatrix{T},
                                C1::Function, C2::ParametrizedMatrix{T}, C3::Function,
                                D1::Function, D2::ParametrizedMatrix{T},
                                H::Function, R::ParametrizedMatrix{T},
                                x1::ParametrizedMatrix{T}, J::AbstractMatrix, S::ParametrizedMatrix{T}) where {T}
        nx, ny, nu, nq, nr, ns = confirm_matrix_sizes(A1(1), A2, A3(1), B1(1), B2, G(1), Q,
                                              C1(1), C2, C3(1), D1(1), D2, H(1), R, x1, J, S)
        new(A1, A2, A3, B1, B2, G, Q, C1, C2, C3, D1, D2, H, R, x1, J, S, nx, ny, nu, nq, nr, ns)
    end

end #ParametrizedSSM

ParametrizedSSM(A2::ParametrizedMatrix{T}, Q::ParametrizedMatrix{T},
                       C2::ParametrizedMatrix{T}, R::ParametrizedMatrix{T},
                       x1::ParametrizedMatrix{T}, S::ParametrizedMatrix{T};
                       A1::Function=_->sparse(1.0I, size(A2,1), size(A2,1)), A3::Function=_->sparse(1.0I, size(A2,2), size(A2,2)),
                       B1::Function=_->sparse(1.0I, size(x1,1), size(x1,1)),
                       B2::ParametrizedMatrix{T}=parametrize_none(spzeros(size(B1(1),2), 1)),
                       G::Function=_->sparse(1.0I, size(x1,1), size(x1,1)),
                       C1::Function=_->sparse(1.0I, size(C2, 1), size(C2, 1)), C3::Function=_->sparse(1.0I, size(C2,2), size(C2,2)),
                       D1::Function=_->sparse(1.0I, size(C1(1),1), size(C1(1),1)),
                       D2::ParametrizedMatrix{T}=parametrize_none(spzeros(size(C1(1),1), 1)),
                       H::Function=_->sparse(1.0I, size(C1(1),1), size(C1(1),1)), J::AbstractMatrix=sparse(1.0I, size(x1, 1), size(x1, 1))) where {T} =
          ParametrizedSSM{T}(A1, A2, A3, B1, B2, G, Q, C1, C2, C3, D1, D2, H, R, x1, J, S)

# State space model parameters
struct SSMParameters{T}

    # Process transition and noise covariance
    A::Vector{T}
    B::Vector{T}
    Q::Vector{T}

    # Observation and noise covariance
    C::Vector{T}
    D::Vector{T}
    R::Vector{T}

    # Initial state and error covariance
    x1::Vector{T}
    S::Vector{T}

end #SSMParameters

SSMParameters(::T; A::Vector{T}=T[], B::Vector{T}=T[], Q::Vector{T}=T[],
               C::Vector{T}=T[], D::Vector{T}=T[], R::Vector{T}=T[],
               x1::Vector{T}=T[], S::Vector{T}=T[]) where {T} =
              SSMParameters{T}(A, B, Q, C, D, R, x1, S)

function (m::ParametrizedSSM{T})(p::SSMParameters{T}) where {T}
    A2, B2, Q = m.A2(p.A), m.B2(p.B), m.Q(p.Q)
    C2, D2, R = m.C2(p.C), m.D2(p.D), m.R(p.R)
    A(t) = m.A1(t) * A2 * m.A3(t)
    B(t) = m.B1(t) * B2
    V(t) = m.G(t) * Q * m.G(t)'
    C(t) = m.C1(t) * C2 * m.C3(t)
    D(t) = m.D1(t) * D2
    W(t) = m.H(t) * R * m.H(t)'
    x1 = vec(m.x1(p.x1))
    P1 = m.J * m.S(p.S) * m.J'
    return StateSpaceModel(A, V, C, W, x1, P1, B=B, D=D)
end #call

function confirm_matrix_sizes(F::AbstractMatrix, B::AbstractMatrix, V::AbstractMatrix,
                              G::AbstractMatrix, D::AbstractMatrix, W::AbstractMatrix,
                              x1::Vector, P1::AbstractMatrix)

    nx = size(F, 1)
    nu = size(B, 2)
    ny = size(G, 1)

    @assert size(F) == (nx, nx)
    @assert size(B) == (nx, nu)
    @assert size(V) == (nx, nx)

    @assert size(G) == (ny, nx)
    @assert size(D) == (ny, nu)
    @assert size(W) == (ny, ny)

    @assert size(x1) == (nx,) || size(x1) == (nx, 1)
    @assert size(P1) == (nx, nx)

    return nx, ny, nu

end #confirm_matrix_sizes

function confirm_matrix_sizes(A1::AbstractMatrix, A2::ParametrizedMatrix{T}, A3::AbstractMatrix,
                           B1::AbstractMatrix, B2::ParametrizedMatrix{T},
                           G::AbstractMatrix, Q::ParametrizedMatrix{T},
                           C1::AbstractMatrix, C2::ParametrizedMatrix{T}, C3::AbstractMatrix,
                           D1::AbstractMatrix, D2::ParametrizedMatrix{T},
                           H::AbstractMatrix, R::ParametrizedMatrix{T},
                           x1::ParametrizedMatrix{T}, J::AbstractMatrix, S::ParametrizedMatrix{T}) where T

    @assert size(B2, 2) == size(D2, 2)

    nx = size(A1, 1)
    ny = size(C1, 1)
    nu = size(B2, 2)

    na1, na2  = size(A1, 2), size(A2, 2)
    nb        = size(B1, 2)
    nc1, nc2  = size(C1, 2), size(C2, 2)
    nd        = size(D1, 2)

    nq = size(Q, 1)
    nr = size(R, 1)
    ns = size(S, 1)

    @assert size(A1) == (nx, na1)
    @assert size(A2) == (na1, na2)
    @assert size(A3) == (na2, nx)

    @assert size(B1) == (nx, nb)
    @assert size(B2) == (nb, nu)

    @assert size(G)  == (nx, nq)
    @assert size(Q)  == (nq, nq)

    @assert size(C1) == (ny, nc1)
    @assert size(C2) == (nc1, nc2)
    @assert size(C3) == (nc2, nx)

    @assert size(D1) == (ny, nd)
    @assert size(D2) == (nd, nu)

    @assert size(H)  == (ny, nr)
    @assert size(R)  == (nr, nr)

    @assert length(x1)  == nx
    @assert size(J)     == (nx, ns)
    @assert size(S)     == (ns, ns)

    return nx, ny, nu, nq, nr, ns

end #confirm_matrix_sizes

"
Generates a realization of a state space model.

Arguments:
  model : StateSpaceModel
  Model defining the process

  n : Int
  Number of steps to simulate.

  u : Array (optional)
  n x nu array of exogenous inputs
"
function simulate(model::StateSpaceModel{T}, n::Int; u::Array{T}=zeros(n, model.nu)) where T
    @assert size(u, 1) == n
    @assert size(u, 2) == model.nu

    # create empty arrays to hold the state and observed series
    x = zeros(model.nx, n)
    y = zeros(model.ny, n)
    u = u'

    # Cholesky decompositions of the covariance matrices, for generating
    # random noise
    V_chol(t) = all(model.V(t) .== 0) ? model.V(t) : cholesky(Hermitian(model.V(t), :L)).U
    W_chol(t) = all(model.W(t) .== 0) ? model.W(t) : cholesky(Hermitian(model.W(t), :L)).U

    # Generate the series
    x[:, 1] = model.x1
    y[:, 1] = model.C(1) * model.x1 + model.D(1) * u[:, 1] + W_chol(1) * randn(model.ny)
    for i=2:n
        x[:, i] = model.A(i-1) * x[:, i-1] + model.B(i-1) * u[:, i-1] + V_chol(i-1) * randn(model.nx)
        y[:, i] = model.C(i)   * x[:, i]   + model.D(i)   * u[:, i]   + W_chol(i) * randn(model.ny)
    end

    return collect(x'), collect(y')
end
