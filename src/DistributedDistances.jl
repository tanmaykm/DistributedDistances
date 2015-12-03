module DistributedDistances

using Distances
using Distances: get_common_ncols, _centralize_colwise
using DistributedArrays

import DistributedArrays: DMatrix, DVector, mappart!
importall Distances

export dcolwise, dcolwise!

sqrt!(d::DArray) = map!(sqrt!, d)

sumsq_percol{T}(d::DMatrix{T}) = DArray(I->sumsq_percol(localpart(d)), size(d, 2), procs(d))
wsumsq_percol{T1, T2}(w::AbstractArray{T1}, d::DMatrix{T2}) = DArray(I->wsumsq_percol(w, localpart(d)), size(d, 2), procs(d))

function dot_percol!(r::DVector, a::DMatrix, b::DMatrix)
    # TODO: check distribution
    mappart!(localr->dot_percol!(localr, localpart(a), localpart(b)), r)
end

function dot_percol(a::DMatrix, b::DMatrix)
    r = DArray(I->zeros(Float64, size(localpart(a),2)), (size(a,2),), procs(a))
    dot_percol!(r, a, b)
end

function dcolwise!(r::DVector, metric::PreMetric, a::AbstractVector, b::DMatrix)
    # TODO: assert proper dimensions and types
    mappart!(localr->colwise!(localr, metric, a, localpart(b)), r)
end

function dcolwise!(r::DVector, metric::PreMetric, a::DMatrix, b::AbstractVector)
    # TODO: assert proper dimensions and types
    mappart!(localr->colwise!(localr, metric, localpart(a), b), r)
end

function dcolwise!(r::DVector, metric::PreMetric, a::DMatrix, b::DMatrix)
    # TODO: assert proper dimensions and types
    mappart!(localr->colwise!(localr, metric, localpart(a), localpart(b)), r)
end

function dcolwise!(r::DVector, metric::SemiMetric, a::DMatrix, b::AbstractVector)
    dcolwise!(r, metric, b, a)
end

function dcolwise(metric::PreMetric, a::DMatrix, b::DMatrix)
    n = get_common_ncols(a, b)
    T = result_type(metric, a, b)
    r = DArray(I->zeros(T, get_common_ncols(localpart(a), localpart(b))), (n,), procs(a))
    dcolwise!(r, metric, a, b)
end

function dcolwise(metric::PreMetric, a::AbstractVector, b::DMatrix)
    n = size(b, 2)
    T = result_type(metric, a, b)
    r = DArray(I->zeros(T, size(localpart(b), 2)), (n,), procs(b))
    dcolwise!(r, metric, a, b)
end

function dcolwise(metric::PreMetric, a::DMatrix, b::AbstractVector)
    n = size(a, 2)
    T = result_type(metric, a, b)
    r = DArray(I->zeros(T, size(localpart(a), 2)), (n,), procs(a))
    dcolwise!(r, metric, a, b)
end

function dcolwise!(r::DVector, dist::CorrDist, a::DMatrix, b::DMatrix)
    dcolwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
end

function dcolwise!(r::DVector, dist::CorrDist, a::AbstractVector, b::DMatrix)
    dcolwise!(r, CosineDist(), _centralize_colwise(a), _centralize_colwise(b))
end

function dcolwise!(r::DVector, dist::WeightedEuclidean, a::DMatrix, b::DMatrix)
    sqrt!(dcolwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end

function dcolwise!(r::DVector, dist::WeightedEuclidean, a::AbstractVector, b::DMatrix)
    sqrt!(dcolwise!(r, WeightedSqEuclidean(dist.weights), a, b))
end

end # module
