using DistributedArrays
using Distances
using DistributedDistances
using Base.Test

macro test_colwise(dist, x, y, xd, yd, tol)
    quote
        local n = size($xd, 2)

        tfx = @timed colwise($dist, $x, $y)
        tdx = @timed dcolwise($dist, $xd, $yd)
        println("\t$(tdx[2]), $(tfx[2])")
        @test_approx_eq_eps tdx[1] tfx[1] $tol

        tfx = @timed colwise($dist, ($x)[:,1], $y)
        tdx = @timed dcolwise($dist, ($x)[:,1], $yd)
        println("\t$(tdx[2]), $(tfx[2])")
        @test_approx_eq_eps tdx[1] tfx[1] $tol

        tfx = @timed colwise($dist, $x, ($y)[:,1])
        tdx = @timed dcolwise($dist, $xd, ($y)[:,1])
        println("\t$(tdx[2]), $(tfx[2])")
        @test_approx_eq_eps tdx[1] tfx[1] $tol
    end
end

macro test_pairwise(dist, x, y, xd, yd, tol)
    quote
        tdx = @timed dpairwise($dist, $x, $yd)
        tfx = @timed pairwise($dist, $x, $y)
        println("\t$(tdx[2]), $(tfx[2])")
        @test_approx_eq_eps tdx[1] tfx[1] $tol
    end
end


# test colwise metrics
function colwise_test()
    println("colwise...")

    m = 5000
    n = 800

    X = rand(m, n);
    Y = rand(m, n);
    A = rand(1:3, m, n);
    B = rand(1:3, m, n);
    P = rand(m, n);
    Q = rand(m, n);
    P[P .< 0.3] = 0.;

    Xd = distribute(X; dist=(1,4));
    Yd = distribute(Y; dist=(1,4));
    Ad = distribute(A; dist=(1,4));
    Bd = distribute(B; dist=(1,4));
    Pd = distribute(P; dist=(1,4));
    Qd = distribute(Q; dist=(1,4));

    println("\tdistributed, singlenode")
    @test_colwise SqEuclidean() X Y Xd Yd eps()
    @test_colwise Euclidean() X Y Xd Yd eps()
    @test_colwise Cityblock() X Y Xd Yd eps()
    @test_colwise Chebyshev() X Y Xd Yd eps()
    @test_colwise Minkowski(2.5) X Y Xd Yd eps()
    @test_colwise Hamming() A B Ad Bd eps()

    @test_colwise CosineDist() X Y Xd Yd eps()
    @test_colwise CorrDist() X Y Xd Yd eps()

    @test_colwise ChiSqDist() X Y Xd Yd eps()
    @test_colwise KLDivergence() P Q Pd Qd eps()
    @test_colwise JSDivergence() P Q Pd Qd eps()
    @test_colwise SpanNormDist() X Y Xd Yd eps()

    @test_colwise BhattacharyyaDist() X Y Xd Yd eps()
    @test_colwise HellingerDist() X Y Xd Yd eps()
    nothing
end


# test pairwise metrics
function pairwise_test()
    println("pairwise...")

    m = 800
    n = 600
    nx = 600
    ny = 600

    X = rand(m, nx)
    Y = rand(m, ny)
    A = rand(1:3, m, nx)
    B = rand(1:3, m, ny)

    P = rand(m, nx)
    Q = rand(m, ny)

    Xd = distribute(X; dist=(1,4));
    Yd = distribute(Y; dist=(1,4));
    Ad = distribute(A; dist=(1,4));
    Bd = distribute(B; dist=(1,4));
    Pd = distribute(P; dist=(1,4));
    Qd = distribute(Q; dist=(1,4));

    println("\tdistributed, singlenode")
    @test_pairwise SqEuclidean() X Y Xd Yd eps()
    @test_pairwise Euclidean() X Y Xd Yd eps()
    @test_pairwise Cityblock() X Y Xd Yd eps()
    @test_pairwise Chebyshev() X Y Xd Yd eps()
    @test_pairwise Minkowski(2.5) X Y Xd Yd eps()
    @test_pairwise Hamming() A B Ad Bd eps()

    @test_pairwise CosineDist() X Y Xd Yd eps()
    @test_pairwise CorrDist() X Y Xd Yd eps()

    @test_pairwise ChiSqDist() X Y Xd Yd eps()
    @test_pairwise KLDivergence() P Q Pd Qd eps()
    @test_pairwise JSDivergence() P Q Pd Qd eps()

    @test_pairwise BhattacharyyaDist() X Y Xd Yd eps()
    @test_pairwise HellingerDist() X Y Xd Yd eps()

    w = rand(m)

    @test_pairwise WeightedSqEuclidean(w) X Y Xd Yd eps()
    @test_pairwise WeightedEuclidean(w) X Y Xd Yd eps()
    @test_pairwise WeightedCityblock(w) X Y Xd Yd eps()
    @test_pairwise WeightedMinkowski(w, 2.5) X Y Xd Yd eps()
    @test_pairwise WeightedHamming(w) A B Ad Bd eps()

    Q = rand(m, m)
    Q = Q * Q'  # make sure Q is positive-definite

    @test_pairwise SqMahalanobis(Q) X Y Xd Yd eps()
    @test_pairwise Mahalanobis(Q) X Y Xd Yd eps()
    nothing
end

colwise_test()
pairwise_test()
