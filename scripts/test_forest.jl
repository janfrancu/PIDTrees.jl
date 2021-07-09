using Revise

using MAT: matread

data_file = "./data/musk.mat"
vars = matread(data_file)

X, y = copy(vars["X"]'), vars["y"]
X, y


using PIDTrees
using StatsBase: fit!, predict


pf = PIDTrees.PIDForest{eltype(X)}()


kwargs = (
		max_depth=3, 
		n_trees=1, 
		max_samples=100, 
		max_buckets=3, 
		epsilon=0.1, 
		threshold=0)


fit!(pf, X; kwargs...)


predict(pf, X)


pf.trees[1].head

for node in pf.trees[1].nodes
	@info("", node.feature, node.children)
end

unique(reduce(vcat, [node.children for node in pf.trees[1].nodes]))

using LightGraphs
g = SimpleDiGraph(pf.trees[1].head);

for (i, node) in enumerate(pf.trees[1].nodes)
	for c in node.children
		add_edge!(g, i, c)
	end
end

using Cairo, GraphPlot, GraphPlot.Compose
draw(PDF("./data/plot/pidtree_fit.pdf", 16cm, 16cm), gplot(g))