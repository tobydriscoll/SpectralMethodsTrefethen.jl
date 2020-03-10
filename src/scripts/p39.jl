# p39.jl - eigenmodes of biharmonic on a square with clamped BCs
#          (compare p38)

# Construct spectral approximation to biharmonic operator:
N = 17
D,x = cheb(N)
D2 = D^2
D2 = D2[2:N,2:N]
S = Diagonal( [0; [1/(1-x^2) for x in x[2:N]]; 0] )
D4 = (Diagonal(1 .- x.^2)*D^4 - 8*Diagonal(x)*D^3 - 12*D^2)*S
D4 = D4[2:N,2:N]
L = kron(I(N-1),D4) + kron(D4,I(N-1)) + 2*kron(D2,I(N-1))*kron(I(N-1),D2)

# Find and plot 25 eigenmodes:
λ,V = eigen(L)
λ,V = real(λ[1:25]),real(V[:,1:25])
λ = sqrt.(λ/λ[1])
y = x
xx = yy = -1:.01:1
sq = [1+1im,-1+1im,-1-1im,1-1im,1+1im]
plt = plot(size=(800,800),layout=(5,5),aspect_ratio=1,framestyle=:none)
for i = 1:25
    U = zeros(N+1,N+1)
    U[2:N,2:N] = reshape(V[:,i],N-1,N-1)
    UU = hcat([ chebinterp(U[:,j]).(yy) for j in 1:N+1 ]...)
    UU = vcat([ chebinterp(UU[i,:]).(xx)' for i in 1:length(yy) ]...)
    contour!(xx,yy,UU,levels=[0],subplot=i,color=:black,
        title=(@sprintf("%.6g",λ[i])),titlefontsize=8 )
    plot!(real(sq),imag(sq),subplot=i)
end
