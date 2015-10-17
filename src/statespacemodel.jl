issquare(x::Matrix) = size(x, 1) == size(x, 2) ? true : false

immutable StateSpaceModel{T}
	# Process transition and noise covariance
	F::Function
	V::Matrix{T}
	# Observation and noise covariance
	G::Function
	W::Matrix{T}
	# Inital guesses at state and error covariance
	x0::Vector{T}
	P0::Matrix{T}

	function StateSpaceModel(F::Function, V::Matrix{T},
	                G::Function, W::Matrix{T}, x0::Vector{T}, P0::Matrix{T})
            F1 = F(1)
            G1 = G(1)
            @assert issquare(F1)
            @assert size(F1, 1) == length(x0)
            @assert issym(V)
            @assert size(V) == size(F1)
            @assert eigmin(V) >= 0
            @assert size(G1, 1) == size(W, 1)
            @assert size(G1, 2) == length(x0)
            @assert issym(W)
            @assert eigmin(W) >= 0
            @assert size(P0, 1) == length(x0)
            @assert issym(P0)
            @assert eigmin(P0) >= 0
            new(F, V, G, W, x0, P0)
	end
end

function StateSpaceModel{T <: Real}(F::Function, V::Matrix{T}, G::Function,
		W::Matrix{T}, x0::Vector{T}, P0::Matrix{T})
	StateSpaceModel{T}(F, V, G, W, x0, P0)
end

function StateSpaceModel{T <: Real}(F::Matrix{T}, V::Matrix{T}, G::Matrix{T},
		W::Matrix{T}, x0::Vector{T}, P0::Matrix{T})
	StateSpaceModel{T}(_->F, V, _->G, W, x0, P0)
end

function show{T}(io::IO, mod::StateSpaceModel{T})
	dx, dy = length(mod.x0), size(mod.G, 1)
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

