# p17.jl - Helmholtz eq. u_xx + u_yy + (k^2)u = f
#         on [-1,1]x[-1,1]    (compare p16.jl)

# Set up spectral grid and tensor product Helmholtz operator:
N = 24
D,x = cheb(N)
y = x
f = [ exp(-10*((y-1)^2+(x-0.5)^2)) for y in y[2:N], x in x[2:N] ]
D2 = D^2
D2 = D2[2:N,2:N]
k = 9
L = kron(I(N-1),D2) + kron(D2,I(N-1)) + k^2*I                     # Laplacian

# Solve for u, reshape to 2D grid, and plot:
u = L\vec(f)
U = zeros(N+1,N+1)
U[2:N,2:N] .= reshape(u,N-1,N-1)
xx = yy = -1:.01:1
UU = hcat([ chebinterp(U[:,j]).(yy) for j in 1:N+1 ]...)
UU = vcat([ chebinterp(UU[i,:]).(xx)' for i in 1:length(yy) ]...)

plt = plot(layout=(1,2))
value = @sprintf("u(0,0) = %0.10e",U[Int(N/2+1),Int(N/2+1)])
surface!(xx,yy,UU,subplot=1,color=:viridis,
  xlabel="x",ylabel="y",zlabel="u",title=value )
contour!(xx,yy,UU,levels=10,fill=true,subplot=2,clims=(-0.02,0.02),color=:balance,
  aspect_ratio=1,xlabel="x",ylabel="y",zlabel="u(x,y)")
