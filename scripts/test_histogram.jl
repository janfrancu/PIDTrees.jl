n = 100
x = round.(Int, randn(100) .* 10)

using StatsBase
counts = countmap(x)

arr = sort(collect(keys(counts)))
count = [counts[a] for a in arr]


using PIDTrees: approx_buckets, best_split


cur_err, b_values = approx_buckets(Float32.(arr), count, 3, 0.1)
sort(collect(b_values[1])) ### just for comparison

num_unique = length(arr)
best_split(cur_err, b_values, 3, num_unique)


"""python
import numpy as np

arr = np.array([-22, -20, -19, -17, -15, -14, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, 
-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 23])
count = np.array([3, 2, 1, 1, 1, 2, 3, 2, 4, 5, 4, 4, 3, 1, 4, 3, 1, 6, 4, 7, 3, 2, 6, 
2, 1, 3, 3, 4, 1, 2, 1, 2, 2, 2, 3, 1, 1])


cur_err, b_values = approx_buckets(arr.astype(np.float32), count, 3, 0.1)


h = Histogram(arr, count, 3, 0.1)
opt, var_red, buckets = h.best_split()


"""


