# p29.jl - solve Poisson equation on the unit disk
#         (compare p16 and p28)
#         with spectral interpolation


# Laplacian in polar coordinates:
N,M = 25,20
N2,M2 = Int((N-1)/2),Int(M/2)
D,r = cheb(N)
D2 = D^2
D1 = D2[2:N2+1,2:N2+1]
D2 = D2[2:N2+1,N:-1:N2+2]
E1 =  D[2:N2+1,2:N2+1]
E2 =  D[2:N2+1,N:-1:N2+2]
dt = 2π/M
t = dt*(1:M)
col = [ 0.5*(-1)^(k+1)/sin(k*dt/2)^2 for k in 1:M-1 ]
D2t = toeplitz([-pi^2/(3*dt^2)-1/6;col])

# Laplacian in polar coordinates:
R = Diagonal(1 ./ r[2:N2+1])
Z = zeros(M2,M2)
L = kron(D1+R*E1,I(M)) + kron(D2+R*E2,[Z I(M2);I(M2) Z]) + kron(R^2,D2t)

# Right-hand side and solution for u:
f = [ -r^2*sin(θ/2)^4 + sin(6θ)*cos(θ/2)^2 for θ in t, r in r[2:N2+1] ]
u = L\vec(f)

# Reshape results onto 2D grid and plot them:
U = reshape(u,M,N2)
rr = LinRange(0,1,80)
tt = [ 2π*m/80 for m in 0:80 ]
U = [zeros(M) U reverse(reverse(U,dims=1),dims=2) zeros(M) ]
UU = vcat([ chebinterp(U[i,:]).(rr)' for i in 1:M ]...)
UU = hcat([ fourinterp(UU[:,j]).(tt) for j in 1:length(rr) ]...)

XX = [ r*cos(θ) for θ in tt, r in rr ]
YY = [ r*sin(θ) for θ in tt, r in rr ]
plt = surface(XX,YY,UU,color=:viridis,cam=(20,40),xaxis="x",yaxis="y",zaxis="u(x,y)")
