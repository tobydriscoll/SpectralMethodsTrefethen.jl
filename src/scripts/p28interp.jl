# p28.jl - eigenmodes of Laplacian on the disk (compare p22)
#    with spectral interpolation

# r coordinate, ranging from -1 to 1 (N must be odd):
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
dt = 2π/M
t = dt*(1:M)
M2 = Int(M/2)
col = [ 0.5*(-1)^(k+1)/sin(k*dt/2)^2 for k in 1:M-1 ]
D2t = toeplitz([-pi^2/(3*dt^2)-1/6;col])

# Laplacian in polar coordinates:
R = Diagonal(1 ./ r[2:N2+1])
Z = zeros(M2,M2)
L = kron(D1+R*E1,I(M)) + kron(D2+R*E2,[Z I(M2);I(M2) Z]) + kron(R^2,D2t)

# Compute four eigenmodes:
index = [1,3,6,10]
λ,V = eigen(-L,sortby=z->abs(z))
λ,V = λ[index],V[:,index]
λ = sqrt.(real(λ/λ[1]))

# Plot eigenmodes:
rr = LinRange(0,1,80)
tt = (0:80)/80*2π
XX = [ r*cos(θ) for θ in tt, r in rr ]
YY = [ r*sin(θ) for θ in tt, r in rr ]
plt = plot(layout=(2,2),size=(600,600),framestyle=:none,color=:balance,clims=(-1,1))
levels = [-1;-0.95:0.1:0.95;1]
for i in 1:4
    str = @sprintf("Mode %d: λ = %0.11f",index[i],λ[i])
    U = reshape(real(V[:,i]),M,N2)
    U = [zeros(M) U reverse(reverse(U,dims=1),dims=2) zeros(M) ]
    UU = vcat([ chebinterp(U[i,:]).(rr)' for i in 1:M ]...)
    UU = hcat([ fourinterp(UU[:,j]).(tt) for j in 1:length(rr) ]...)
    UU = UU/norm(UU,Inf)
    contour!(XX,YY,UU,levels=levels,color=:balance,fill=true,subplot=i)
    contour!(XX,YY,UU,levels=[0],subplot=i,color=:black,linewidth=1.5)
    plot!(cos.(tt),sin.(tt),l=(:black,1.5),subplot=i,title=str)
end
