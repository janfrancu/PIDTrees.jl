import StatsBase: predict, fit!
using StatsBase: sample, Weights, quantile


struct PIDNode{T}
    feature::Int
    splits::Array{T}
    children::Array{Int}
    hc::HyperCube{T}
end

mutable struct PIDTree{T}
	head::Int
	nodes::Array{PIDNode{T}}
	densities::Array{T}

	PIDTree{T}() where {T} = new(0, PIDNode[], T[])
end

mutable struct PIDForest{T}
	trees::Array{PIDTree{T}}

	PIDForest{T}() where {T} = new(PIDTree{T}[])
end

### reserves space in the node array from head to head+n
function reserve_children!(pt::PIDTree{T}, n) where {T}
	append!(pt.nodes, fill(PIDNode{T}(), n))
	children = pt.head .+ collect(1:n)
    pt.head += n
    children
end

### shortcut for defining leaf node
PIDNode{T}(h::HyperCube{T}) where {T} = PIDNode(-1, T[], Int[], h)
PIDNode{T}() where {T} = PIDNode(-1, T[], Int[], HyperCube{T}())

### head tracks the preallocated position for the node in the node array

function grow!(pt::PIDTree{T}, X::AbstractArray{T, 2}, indeces, h::HyperCube{T}, current_depth, head, kwargs) where {T}
	if current_depth < kwargs.max_depth && length(indeces) > 1
		# find the best feature to split the data into
		valid_dims = findall(minimum(X[:, indeces], dims=2)[:] .!= maximum(X[:, indeces], dims=2)[:])
		
		best_splits = map(valid_dims) do d
			vals, counts = value_counts(X[d, indeces])
			gaps = optimal_gaps(vals, h.low[d], h.high[d])
			cur_err, b_values = approx_buckets(gaps./counts, counts, kwargs.max_buckets, kwargs.epsilon)
			num_unique = length(vals)
			_, var_red, buckets = best_split(cur_err, b_values, 3, num_unique)
			var_red, buckets, vals
		end

		scores = map(x -> x[1], best_splits)

		# scores did not get above threshold, make an empty leaf node with some volume
		if all(scores .<= kwargs.threshold)
			pt.nodes[head] = PIDNode{T}(h)
		else
			# sample the split with weights
			sdi = sample(1:length(valid_dims), Weights(scores./sum(scores)))
			split_dim = valid_dims[sdi]
			buckets = best_splits[sdi][2]
			vals = best_splits[sdi][3]
			split_vals = [(vals[i - 1] + vals[i]) / 2 for i in buckets]
			
			num_children = length(split_vals) + 1
			# extends the node array by undefined pointers in range(pt.head, pt.head + length(splits))
			children = reserve_children!(pt, num_children)
			pt.nodes[head] = PIDNode(split_dim, split_vals, children, h)

			# found the splits, try to grow further in each bucket
			for (i, c) in enumerate(children)
				# limit hypecubes based on split
				if 1 < i < num_children
					hh = restrict(h, split_dim, split_vals[i-1], split_vals[i])
				elseif i == 1
					hh = restrict(h, split_dim, h.low[split_dim], split_vals[1])
				else 
					hh = restrict(h, split_dim, split_vals[end], h.high[split_dim])
				end

				# limit indeces based restricted hyper cubes
				new_indeces = restrict_indices(X, indeces, hh, split_dim)

				grow!(pt, X, new_indeces, hh, current_depth + 1, c, kwargs)
			end
		end
	else
		pt.nodes[head] = PIDNode{T}(h)
	end
	pt
end


### replaces empty array density with the correct values by computing volume and number of samples that got to each leaf
### when we know that a set of samples will end in some empty node we just compute its
### volume and corresponding number of samples in that volume and make the density est.
### at that time we know the 
function compute_density!(pt::PIDTree{T}, X::AbstractArray{T, 2}, indeces) where {T}
	densities = map(pt.nodes) do node
		new_indeces = restrict_indices(X, indeces, node.hc)
		num_samples = length(new_indeces)
		num_samples == 0 ? zero(T) : T(log(num_samples)) - volume(node.hc)
	end
	pt.densities = densities
	pt
end


function fit!(pf::PIDForest{T}, X::AbstractArray{T, 2}; n_trees::Int=50, max_samples=100, max_depth=10,
												max_buckets=3, epsilon=0.1, threshold=0.0) where {T}
	D, N = size(X)
	idx_density = sample(1:N, min(max_depth * 200, N); replace=false) # these samples are used for density computation
	
	trees = map(1:n_trees) do i
		idx_train = sample(1:N, min(max_samples, N); replace=false) # these samples are used for training individual trees
		h = HyperCube(X[:, idx_train])
		pt = PIDTree{T}()
		head = reserve_children!(pt::PIDTree, 1)[1] # extends the array by one for the root node
		pt = grow!(pt, X, idx_train, h, 0, head,
					(max_depth=max_depth, 
						threshold=threshold,
						max_buckets=max_buckets,
						epsilon=epsilon)
					)

		compute_density!(pt, X, idx_density)
	end

	pf.trees = trees
	pf
end


# finds which indeces correspond to each bucket
function split!(split_indices, pt::PIDTree, node_idx::Int, X, indeces)
	node = pt.nodes[node_idx]
	# unless in leaf node continue split or 
	if length(node.children) > 0
		for c in node.children
			cnode = pt.nodes[c]
			new_indeces = restrict_indices(X, indeces, cnode.hc, node.feature)
			if length(new_indeces) > 0
				split!(split_indices, pt, c, X, new_indeces)
			end
		end
	else
		split_indices[indeces] .= node_idx
	end
end

function predict(pt::PIDTree, X)
	# split indeces should be a list of pointers to pt.nodes/pt.densities arrays
	# indicating where each sample has ended in its path through the tree
	# it may happen that some samples do not belong to any leaf -> 0.0
	num_samples = size(X, 2)
	split_indices = zeros(Int, num_samples)
	split!(split_indices, pt, 1, X, collect(1:num_samples))
	map(split_indices) do si
		(si > 0) ? pt.densities[si] : zero(eltype(pt.densities))
	end
end

function predict(pf::PIDForest, X; pct=75)
	scores = map(pf.trees) do pt
		predict(pt, X)
	end
	scores = reduce(hcat, scores)
	mapslices(s -> quantile(s, pct/100.0), scores; dims=2)[:]
end