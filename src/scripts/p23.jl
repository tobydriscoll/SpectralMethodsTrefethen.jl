# p23.jl - eigenvalues of perturbed Laplacian on [-1,1]x[-1,1]
#         (compare p16.jl)

# Set up tensor product Laplacian and compute 4 eigenmodes:
N = 16
D,x = cheb(N)
y = x

D2 = D^2
D2 = D2[2:N,2:N]
L = -kron(I(N-1),D2) - kron(D2,I(N-1))                    #  Laplacian
f = [ exp(20*(y-x-1)) for y in y[2:N], x in x[2:N] ]      # perturbation
L = L + Diagonal(vec(f))
λ,V = eigen(L)
λ,V = λ[1:4],V[:,1:4]

# Reshape them to 2D grid, interpolate to finer grid, and plot:
xx = yy = -1:.02:1
U = zeros(N+1,N+1)
plt = plot(layout=(2,2))
for i in 1:4
    global U
    U[2:N,2:N] = reshape(V[:,i],N-1,N-1)
    UU = hcat([ chebinterp(U[:,j]).(yy) for j in 1:N+1 ]...)
    UU = vcat([ chebinterp(UU[i,:]).(xx)' for i in 1:length(yy) ]...)
    m,M = extrema(UU)
    scl = abs(m) > M ? m : M
    UU = UU/scl  
    str = @sprintf("λ = %0.12f π^2/4",λ[i]/(π^2/4)) 
    contour!(xx,yy,UU,fill=true,subplot=i,levels=-1.1:.2:1.1,
      color=:balance,clims=(-1,1),
      xlim=(-1,1),ylim=(-1,1),aspect_ratio=1,title=str)
end
