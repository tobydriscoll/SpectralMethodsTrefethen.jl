# p24.jl - pseudospectra of Davies's complex harmonic oscillator
#         (For finer, slower plot, change 0:2 to 0:.5.)

# Eigenvalues:
N = 70; (D,x) = cheb(N); x = x[2:N];
L = 6; x = L*x; D = D/L;                   # rescale to [-L,L]
A = -D^2; A = A[2:N,2:N] + (1+3im)*diagm(x.^2);
lambda = eigvals(A);
clf(); plot(real(lambda),imag(lambda),"k.",markersize=6);
axis([0,50,0,40])

# Pseudospectra:
x = 0:1:50; y = 0:1:40;  zz = x' .+ 1im*y;
I = eye(N-1);
minsvd(z) = minimum(svdvals(z*I-A));
sigmin = [ minsvd(x[i]+1im*y[j]) for i=eachindex(x), j=eachindex(y) ];
contour(x,y,sigmin',levels=10.0.^(-4:.5:-.5));
