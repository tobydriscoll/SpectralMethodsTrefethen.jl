# p36.m - Laplace eq. on [-1,1]x[-1,1] with nonzero BCs

# Set up grid and 2D Laplacian, boundary points included:
N = 24
D,x = cheb(N)
y = x
D2 = D^2
L = kron(I(N+1),D2) + kron(D2,I(N+1))

# Impose boundary conditions by replacing appropriate rows of L:
isboundary = (x,y) -> (abs(x)==1 || abs(y)==1)
b = vec([isboundary(x,y) for y in y, x in x ])       # boundary pts
L[b,:] .= 0
L[b,b] = I(4N)
rhs = zeros((N+1)^2)
rhs[b] = [ (y==1)*(x<0)*sin(π*x)^4 + 0.2*(x==1)*sin(3π*y) for y in y, x in x if isboundary(x,y) ]

# Solve Laplace equation, reshape to 2D, and plot:
u = L\rhs
U = reshape(u,N+1,N+1)
xx = yy = -1:.04:1;
s = Spline2D(reverse(x),reverse(y),reverse(reverse(U,dims=2),dims=1))
plt = surface(xx,yy,evalgrid(s,xx,yy),color=:viridis,
  xaxis="x",yaxis="y",zaxis=("u(x,y)"),cam=(-20,45))
umid = U[Int(N/2)+1,Int(N/2)+1]
title!( @sprintf("u(0,0) = %0.9f",umid) )
