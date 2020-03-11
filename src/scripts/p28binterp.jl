# p28b.jl - eigenmodes of Laplacian on the disk
#           with spectral interpolation

# r coordinate, ranging from -1 to 1 (N must be odd)
N = 25
N2 = Int((N-1)/2)
D,r = cheb(N)
D2 = D^2
D1 = D2[2:N2+1,2:N2+1]
D2 = D2[2:N2+1,N:-1:N2+2]
E1 =  D[2:N2+1,2:N2+1]
E2 =  D[2:N2+1,N:-1:N2+2]

# t = theta coordinate, ranging from 0 to 2*pi (M must be even):
M = 20
dt = 2*pi/M
t = dt*(1:M)
M2 = Int(M/2)
col = [ 0.5*(-1)^(k+1)/sin(k*dt/2)^2 for k in 1:M-1 ]
D2t = toeplitz([-pi^2/(3*dt^2)-1/6;col])

# Laplacian in polar coordinates:
R = Diagonal(1 ./ r[2:N2+1])
Z = zeros(M2,M2)
L = kron(D1+R*E1,I(M)) + kron(D2+R*E2,[Z I(M2);I(M2) Z]) + kron(R^2,D2t)

# Compute 25 eigenmodes:
index = 1:25
λ,V = eigen(-L,sortby=z->abs(z))
λ,V = λ[index],V[:,index]
λ = sqrt.(real(λ/λ[1]))

# Plot nodal lines:
rr = LinRange(0,1,80)
tt = (0:80)/80*2π
XX = [ r*cos(θ) for θ in tt, r in rr ]
YY = [ r*sin(θ) for θ in tt, r in rr ]
plt = plot(size=(800,800),layout=(5,5),aspect_ratio=1,framestyle=:none)
for i = 1:25
    str = @sprintf("%0.4f",λ[i])
    plot!(cos.(tt),sin.(tt),l=(:black,1.5),subplot=i)
    plot!(cos.(tt),sin.(tt),l=(:black,2),subplot=i,
      title=str,titlefontsize=8,xlim=(-1.04,1.04),ylim=(-1.04,1.04))
    U = reshape(real(V[:,i]),M,N2)
    U = [zeros(M) U reverse(reverse(U,dims=1),dims=2) zeros(M) ]
    UU = vcat([ chebinterp(U[i,:]).(rr)' for i in 1:M ]...)
    UU = hcat([ fourinterp(UU[:,j]).(tt) for j in 1:length(rr) ]...)
    contour!(XX,YY,UU,levels=[0],color=:black,subplot=i)
end
