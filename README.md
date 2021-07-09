# PIDTrees.jl
This is a reimplementation of the PIDForest[1] anomaly detection method in the Julia programming language. It does not differ much from the original Python at ![https://github.com/vatsalsharan/pidforest](https://github.com/vatsalsharan/pidforest), however it runs on average 8 times faster straight out of the box.

[1] Gopalan, Parikshit and Sharan, Vatsal and Wieder, Udi. *PIDForest: Anomaly Detection via Partial Identification* Advances in Neural Information Processing Systems. 2019.

## Result replication
In order to replicate the results the easiest way is to activate julia environment defined in `./scripts/` folder.
```bash
    julia --project=scripts
```
Once julia is started run the following to instantiate the environment.
```julia
] instantiate
```
This will download all the pkg used to visualize and evaluate the method's results.
The script for replication is located in `./scripts/test_end2end.jl` and can be run in the following way
```bash
    julia --project=scripts ./scripts/test_end2end.jl
```
and should produce similar table to this
```
     │ dataset       auc      fit_t    eval_t   auc_std  fit_t_std  eval_t_std 
    ─┼────────────────────────────────────────────────────────────────────────
     │ thyroid        0.873    0.713    0.066    0.013      0.025       0.004
     │ mammography    0.845    0.851    0.087    0.005      0.013       0.001
     │ satimage-2     0.985    1.685    0.058    0.002      0.03        0.004
     │ vowels         0.746    1.24     0.017    0.008      0.033       0.004
     │ siesmic        0.733    1.519    0.027    0.008      0.023       0.004
     │ musk           1.0     23.711    0.041    0.0        2.058       0.002
     │ http           0.989    1.775    6.969    0.003      0.137       0.457
     │ smtp           0.923    0.697    0.825    0.007      0.011       0.016
```


## Instalation for use outside of this repo
Installing into a new environment with one command
```julia
] add https://github.com/janfrancu/PIDTrees.jl
```
Features matrix should have samples stored in columns.
```jl
    using PIDTrees: PIDForest
    using StatsBase: predict, fit!
    using EvalMetrics # not necessary, used for evaluation

    X, y = ... # load data

    kwargs = kwargs = (
                max_depth=10, 
                n_trees=50, 
                max_samples=100, 
                max_buckets=3, 
                epsilon=0.1, 
                threshold=0)

    pf = PIDForest{eltype(X)}() # initializes an empty forest
    fit!(pf, X; kwargs...)
    scores = predict(pf, X; pct=50)

    roc = EvalMetrics.roccurve(y, -scores) # have to be negated because scores correspond to densities (low density is anomalous)
    auc = EvalMetrics.auc_trapezoidal(roc...)
```

## Known issues
- testing on separate data leads gets some samples into regions of no training samples
- cannot train on dataset with some features being constant value (happens quite often when training on clean data)
- may be still too slow for some application