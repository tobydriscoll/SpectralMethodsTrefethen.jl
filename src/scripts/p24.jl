# p24.jl - pseudospectra of Davies's complex harmonic oscillator

# Eigenvalues:
N = 70
D,x = cheb(N)
L = 6
x,D = L*x,D/L            # rescale to [-L,L]
A = -D^2 + (1+3im)*Diagonal(x.^2)
A = A[2:N,2:N]
λ = eigvals(A)
plt = scatter(real(λ),imag(λ),xlim=(0,50),ylim=(0,40),aspect_ratio=1)

# Pseudospectra:
x,y = 0:0.5:50,0:0.5:40
minsvd = z -> minimum(svdvals(z*I-A))
sigmin = [ minsvd(x+1im*y) for y in y, x in x ]
contour!(x,y,log10.(sigmin),levels=-4:0.5:-0.5,color=:viridis)
