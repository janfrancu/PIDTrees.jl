using StatsBase: countmap

"""
    arr is assumed to be sorted and unique
"""
function approx_buckets(arr::AbstractArray{T}, count, max_buckets, eps) where {T}
    err_a = zeros(T, max_buckets) .- T(1)
    cur_err = zeros(T, max_buckets)
    b_values = [Dict{Int, Tuple{Int,T,T,T,Int}}() for _ in 1:max_buckets]
    cur_sum = T(0)
    cur_sq = T(0)
    cur_pts = 0
    for j in 1:length(arr)
        cur_sum += arr[j] * count[j]
        cur_sq += (arr[j] * arr[j]) * count[j]
        cur_pts += count[j]
        cur_err[1] = cur_sq - cur_sum * cur_sum / cur_pts
        if cur_err[1] > (1 + eps) * err_a[1]
            err_a[1] = cur_err[1]
        else
            delete!(b_values[1], j - 1)
        end
        b_values[1][j] = (0, cur_err[1], cur_sum, cur_sq, cur_pts)
        for k in 2:max_buckets
            cur_err[k] = cur_err[k - 1]
            a_val = j + 1
            for b_val in keys(b_values[k - 1])
                if b_val < j
                    _, b_err, b_sum, b_sq, b_pts = b_values[k - 1][b_val]
                    tmp_error = b_err + cur_sq - b_sq - (cur_sum - b_sum)^2 / (cur_pts - b_pts)
                    if tmp_error < cur_err[k]
                        cur_err[k] = tmp_error
                        a_val = b_val + 1
                    end
                end
            end
            b_values[k][j] = (a_val, cur_err[k], cur_sum, cur_sq, cur_pts)
            if cur_err[k] > (1 + eps) * err_a[k]
                err_a[k] = cur_err[k]
            else
                delete!(b_values[k], j - 1)
            end
        end
    end
    cur_err, b_values
end

function compute_buckets(opt, b_values, num_buckets, num_unique)
    buckets = []
    e = num_unique
    k = num_buckets
    while e >= 0
        start = b_values[k][e][1]
        if start <= e
            push!(buckets, start)
        end
        e = start - 1
        k -= 1
    end
    reverse(buckets)
end

function best_split(err, b_values, max_buckets, num_unique)
    if err[1] ≈ 0.0
        return 0, 0, []
    else
        err_red = [(err[1] - err[i]) for i in 2:max_buckets]
        var_red = maximum(err_red) / err[1]
        if var_red < 0
            @error("var_red is", var_red)
            var_red = 0
        end
        opt = argmax(err_red) + 2
        buckets = compute_buckets(opt, b_values, max_buckets, num_unique)
        
        return opt, var_red, buckets[2:end]
    end
end

"""
 this could be faster
 aggregation over rows in col major will make definitely an impact
 furthermore we are using only as subset of samples -> "array view"
 doing it manually over all dimensions is definitely the way to go
"""
function value_counts(x)
    counts = countmap(x)
    vals = sort(collect(keys(counts)))
    vals, [counts[a] for a in vals]
end

"""
    function optimal_gaps(vals::AbstractArray{T}, s::T, e::T) where {T}

Given an array of sorted unique values `vals` and bounding interval (s, e), 
this function computes middle points of each consecutive pair of points in `vals`.
"""
function optimal_gaps(vals::AbstractArray{T}, s::T, e::T) where {T}
    num_unique = length(vals)
    if num_unique <= 1
        return zeros(T, 1)
    else
        gaps = similar(vals, num_unique)
        gaps[1] = (vals[1] + vals[2]) / 2 - s
        gaps[end] = e - (vals[end] + vals[end-1]) / 2
        for i in 2:num_unique-1
            # same as (vals[i + 1] + vals[i]) / 2 - (vals[i - 1] + vals[i]) / 2
            gaps[i] = (vals[i + 1] - vals[i - 1]) / 2
        end
        return gaps
    end
end