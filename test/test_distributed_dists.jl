using DistributedArrays
using Distances
using DistributedDistances
using Base.Test

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

macro test_colwise(dist, x, y, xd, yd, tol)
    quote
        local n = size($xd, 2)
        r1 = zeros(n)
        r2 = zeros(n)
        r3 = zeros(n)
        for j = 1 : n
            r1[j] = evaluate($dist, ($x)[:,j], ($y)[:,j])
            r2[j] = evaluate($dist, ($x)[:,1], ($y)[:,j])
            r3[j] = evaluate($dist, ($x)[:,j], ($y)[:,1])
        end
        @test_approx_eq_eps dcolwise($dist, $xd, $yd) r1 $tol
        @test_approx_eq_eps dcolwise($dist, ($x)[:,1], $yd) r2 $tol
        @test_approx_eq_eps dcolwise($dist, $xd, ($y)[:,1]) r3 $tol
    end
end

@test_colwise SqEuclidean() X Y Xd Yd 1.0e-12
@test_colwise Euclidean() X Y Xd Yd 1.0e-12
@test_colwise Cityblock() X Y Xd Yd 1.0e-11
@test_colwise Chebyshev() X Y Xd Yd 1.0e-16
@test_colwise Minkowski(2.5) X Y Xd Yd 1.0e-13
@test_colwise Hamming() A B Ad Bd 1.0e-16

@test_colwise CosineDist() X Y Xd Yd 1.0e-12
@test_colwise CorrDist() X Y Xd Yd 1.0e-12

@test_colwise ChiSqDist() X Y Xd Yd 1.0e-12
@test_colwise KLDivergence() P Q Pd Qd 1.0e-13
@test_colwise JSDivergence() P Q Pd Qd 1.0e-13
@test_colwise SpanNormDist() X Y Xd Yd 1.0e-12

@test_colwise BhattacharyyaDist() X Y Xd Yd 1.0e-12
@test_colwise HellingerDist() X Y Xd Yd 1.0e-12

