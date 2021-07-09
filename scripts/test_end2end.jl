using Revise

using MAT: matread
using PIDTrees
using StatsBase: fit!, predict
using EvalMetrics
using DataFrames
using Statistics

results = []
for dataset in ["thyroid","mammography","satimage-2","vowels","siesmic","musk","http","smtp"]
	for seed in 1:5
		data_file = "./data/$(dataset).mat"
		vars = matread(data_file)
		X, y = copy(vars["X"]'), vars["y"][:]
		D, N = size(X)

		pf = PIDTrees.PIDForest{eltype(X)}()

		kwargs = (
				max_depth=10, 
				n_trees=50, 
				max_samples=100, 
				max_buckets=3, 
				epsilon=0.1, 
				threshold=0)

		try
			_, fit_t, _, _, _ = @timed fit!(pf, X; init_seed=seed, kwargs...)
			scores, eval_t, _, _, _ = @timed predict(pf, X; pct=50)

			roc = EvalMetrics.roccurve(y, -scores)
			auc = EvalMetrics.auc_trapezoidal(roc...)

			@info("SUCCESS", dataset, seed, auc, fit_t, eval_t)
			push!(results, DataFrame([(;dataset=dataset, auc=auc, fit_t=fit_t, eval_t=eval_t, N=N, D=D)]))
		catch e
			@warn("FAILURE", dataset, e)
		end
	end
end

df = reduce(vcat, results)
metrics = [:auc, :fit_t, :eval_t]
df_agg = combine(groupby(df, :dataset), metrics .=> mean .=> metrics, metrics .=> std)
df_agg[:, 2:end] .= round.(df_agg[:, 2:end], digits=3)
@info df_agg