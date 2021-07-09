struct HyperCube{T}
	low::Vector{T}
	high::Vector{T}
end

HyperCube{T}() where {T} = HyperCube(T[], T[])

function HyperCube(X::AbstractArray{T, 2}) where {T<:Real}
	D, N = size(X)
	low, high = zeros(T, D), zeros(T, D)
	for d in 1:D
		val = sort(unique(X[d,:]))
		@assert length(val) > 1 "No entropy in dimension $d"
		low[d] = (T(3) * val[1] - val[2]) / T(2)  		# v[1] + (v[1] - v[2])/2
		high[d] = (T(3) * val[end] - val[end-1]) / T(2)	# v[end] + (v[end] - v[end-1])/2
	end
	HyperCube{T}(low, high)
end

volume(h::HyperCube{T}) where {T<:Real} = sum(log, h.high .- h.low)

function restrict(h::HyperCube{T}, feature_idx::Int, low::T, high::T) where {T<:Real}
	@assert low < high
	hh = deepcopy(h)
	hh.low[feature_idx] = low
	hh.high[feature_idx] = high
	hh
end

Base.length(h::HyperCube{T}) where {T} = length(h.low)

function restrict_indices(X, indeces, h::HyperCube{T}, d::Int) where {T}
	mask = h.low[d] .<= X[d, indeces] .< h.high[d]
	indeces[mask]
end

function restrict_indices(X, indeces, h::HyperCube{T}) where {T}
	mask_all = h.low .<= X[:, indeces] .< h.high
	mask = [all(column) for column in eachcol(mask_all)]
	indeces[mask]
end