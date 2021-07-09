using PIDForest
using Test


### real data
# include("./scripts/test_data.jl")

### simulated data
X = reshape(Float32.(collect(1:8)), (2,4))
h = HyperCube(X)

@test all(h.low .< X)
@test all(h.high .> X)
@test PIDForest.volume(h) ≈ log(8) + log(8)

hh = PIDForest.restrict(h, 2, 3.0f0, 7.0f0)
@test PIDForest.volume(hh) ≈ log(8) + log(4)

