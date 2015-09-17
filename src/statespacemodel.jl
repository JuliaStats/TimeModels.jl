abstract AbstractStateSpaceModel

function validate_model_elements(B, U, Q, Z, A, R, x1, V1)

    nx = size(B, 1)
    ny = size(Z, 1)
    nu = size(U, 2)

    @assert size(B, 1) == nx
    @assert size(B, 2) == nx
    @assert size(U, 1) == nx
    @assert size(Q, 1) == nx

    @assert size(Z, 1) == ny
    @assert size(Z, 2) == nx
    @assert size(A, 1) == ny
    @assert size(R, 1) == ny

    @assert length(x1) == nx
    @assert size(x1, 1) == nx
    @assert size(V1, 1) == nx
    @assert size(V1, 1) == nx

    @assert nu == size(A, 2)

    return nx, ny, nu

end #validate_model_elements

immutable StateSpaceModel{T<:Real} <: AbstractStateSpaceModel
    # Process transition and noise covariance
    B::Matrix{T}
    U::Matrix{T}
    Q::Matrix{T}
    # Observation and noise covariance
    Z::Matrix{T}
    A::Matrix{T}
    R::Matrix{T}
    # Inital guesses at state and error covariance
    x1::Array{T}
    V1::Matrix{T}

    nx::Int
    ny::Int
    nu::Int

    function StateSpaceModel(B::Matrix{T}, U::Matrix{T}, Q::Matrix{T}, Z::Matrix{T}, A::Matrix{T}, R::Matrix{T},
                x1::Array{T}, V1::Matrix{T})

        ispossemidef(x::Matrix) = issym(x) && (eigmin(x) >= 0)
        @assert ispossemidef(Q)
        @assert ispossemidef(R)
        @assert ispossemidef(V1)

        nx, ny, nu = validate_model_elements(B, U, Q, Z, A, R, x1, V1)
        new(B, U, Q, Z, A, R, x1, V1, nx, ny, nu)
    end
end

function StateSpaceModel{T<:Real}(
        B::Matrix{T}, U::Matrix{T}, Q::Matrix{T},
        Z::Matrix{T}, A::Matrix{T}, R::Matrix{T},
        x1::Array{T}, V1::Matrix{T})

	  StateSpaceModel{T}(B, U, Q, Z, A, R, x1, V1)
end

function StateSpaceModel{T<:Real}(
        B::Matrix{T}, Q::Matrix{T},
        Z::Matrix{T}, R::Matrix{T},
        x1::Array{T}, V1::Matrix{T})

	  StateSpaceModel{T}(B, zeros(size(B,1),1), Q, Z, zeros(size(Z,1),1), R, x1, V1)
end

immutable ParametrizedMatrix{T<:Real}

    f::Vector{T}
    D::Matrix{T}
    np::Int
    dims::Tuple{Int, Int}

    function ParametrizedMatrix(f::Vector{T}, D::Matrix{T}, dims::Tuple{Int, Int})
        @assert length(f) == size(D, 1)
        @assert length(f) == dims[1] * dims[2]
        new(f, D, size(D, 2), dims)
    end #ParametrizedMatrix

end #ParametrizedMatrix

ParametrizedMatrix{T}(f::Vector{T}, D::Matrix{T}, dims::Tuple{Int, Int}) = ParametrizedMatrix{T}(f, D, dims)

function show{T}(io::IO, cpm::ParametrizedMatrix{T})

    combinestrelems(a, b) = ((a != "") & (b != "")) ?
        a * " + " * b : a * b

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
        paramstring[i] = reduce(combinestrelems, paramelems)
    end #for

    finalstrings = map(combinestrelems, conststring, paramstring)

    showcompact(reshape(finalstrings, cpm.dims))
end

length(cpm::ParametrizedMatrix) = cpm.dims[1] * cpm.dims[2]
size(cpm::ParametrizedMatrix) = cpm.dims
size(cpm::ParametrizedMatrix, dim::Int) = cpm.dims[dim]
call(cpm::ParametrizedMatrix, params::Vector) = all(cpm.D .== 0) ?
        reshape(cpm.f, cpm.dims) : reshape(cpm.f + cpm.D * params, cpm.dims)

parametrize_full(X::Union{Vector, Matrix}) =
        ParametrizedMatrix(zeros(length(X)), eye(length(X)), size(X'')), collect(X)

function parametrize_diag(X::Matrix)
    n = size(X, 1)
    D = zeros(n^3)
    D[[1 + (i-1)*(n^2 + n + 1) for i in 1:n]] = 1
    D = reshape(D, n^2, n)
    #f = collect((eye(n) .== 0) .* X) #TODO: Revisit choice of default behaviour here?
    f = zeros(n^2)
    return ParametrizedMatrix(f, D, (n,n)), diag(X)
end #parametrize_diag

parametrize_none{T}(X::Union{Vector{T}, Matrix{T}}) =
        ParametrizedMatrix(collect(X), zeros(length(X))'', size(X'')), T[] 

immutable ParametrizedSSM <: AbstractStateSpaceModel

    # Process transition and noise covariance
    B::ParametrizedMatrix
    U::ParametrizedMatrix
    Q::ParametrizedMatrix

    # Observation and noise covariance
    Z::ParametrizedMatrix
    A::ParametrizedMatrix
    R::ParametrizedMatrix

    # Initial state and error covariance
    x1::ParametrizedMatrix
    V1::ParametrizedMatrix

    nx::Int
    ny::Int
    nu::Int

    function ParametrizedSSM(B::ParametrizedMatrix, U::ParametrizedMatrix, Q::ParametrizedMatrix,
                Z::ParametrizedMatrix, A::ParametrizedMatrix, R::ParametrizedMatrix,
                x1::ParametrizedMatrix, V1::ParametrizedMatrix)
        nx, ny, nu = validate_model_elements(B, U, Q, Z, A, R, x1, V1)
        new(B, U, Q, Z, A, R, x1, V1, nx, ny, nu)
    end

end #ParametrizedSSM

function ParametrizedSSM(B::ParametrizedMatrix, Q::ParametrizedMatrix,
            Z::ParametrizedMatrix, R::ParametrizedMatrix,
            x1::ParametrizedMatrix, V1::ParametrizedMatrix)
    U = parametrize_none(zeros(size(B,1), 1))[1]
    A = parametrize_none(zeros(size(Z,1), 1))[1]
    ParametrizedSSM(B, U, Q, Z, A, R, x1, V1)
end

immutable SSMParameters{T}

    # Process transition and noise covariance
    B::Vector{T}
    U::Vector{T}
    Q::Vector{T}

    # Observation and noise covariance
    Z::Vector{T}
    A::Vector{T}
    R::Vector{T}

    # Initial state and error covariance
    x1::Vector{T}
    V1::Vector{T}

end #SSMParameters

SSMParameters{T}(B::Vector{T}, Q::Vector{T}, Z::Vector{T}, R::Vector{T},
              x1::Vector{T}, V1::Vector{T}) = SSMParameters(B, T[], Q, Z, T[], R, x1, V1)

call(m::ParametrizedSSM, p::SSMParameters) = StateSpaceModel(m.B(p.B), m.U(p.U), m.Q(p.Q),
                                                m.Z(p.Z), m.A(p.A), m.R(p.R), m.x1(p.x1), m.V1(p.V1))

