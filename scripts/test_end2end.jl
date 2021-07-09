using Revise

using MAT: matread

data_file = "./data/musk.mat"
vars = matread(data_file)

X, y = copy(vars["X"]'), vars["y"][:]
X, y


using PIDTrees
using StatsBase: fit!, predict


pf = PIDTrees.PIDForest{eltype(X)}()


kwargs = (
		max_depth=10, 
		n_trees=50, 
		max_samples=100, 
		max_buckets=3, 
		epsilon=0.1, 
		threshold=0)


fit!(pf, X; kwargs...)

scores = predict(pf, X; pct=50)


using EvalMetrics
roc = EvalMetrics.roccurve(y, -scores)
auc = EvalMetrics.auc_trapezoidal(roc...)