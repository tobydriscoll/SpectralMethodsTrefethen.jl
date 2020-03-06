# p16.jl - Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary

# Set up grids and tensor product Laplacian and solve for u:
N = 24
D,x = cheb(N)
y = x
f = [ 10*sin(8x*(y-1)) for y in y[2:N], x in x[2:N] ]
D2 = D^2
D2 = D2[2:N,2:N]
L = kron(I(N-1),D2) + kron(D2,I(N-1))                       # Laplacian

plt = plot(layout=(1,2))
spy!(sparse(L),subplot=1,title="Nonzeros in the Laplacian")
@time u = L\vec(f)           # solve problem and watch the clock

# Reshape long 1D results onto 2D grid (reversing to usual direction):
U = zeros(N+1,N+1)
U[N:-1:2,N:-1:2] .= reshape(u,N-1,N-1)
value = U[Int(3N/4+1),Int(3N/4+1)]

# Interpolate to finer grid and plot:
xx = yy = -1:.01:1
UU = evalgrid(Spline2D(reverse(y),reverse(x),U),yy,xx)
str = latexstring("u(2^{-1/2},2^{-1/2}) = "*@sprintf("%0.11f",value))
surface!(xx,yy,UU,subplot=2,color=:balance,clims=(-.5,.5),
  xlabel="x",ylabel="y",zlabel="u(x,y)",title=str)

