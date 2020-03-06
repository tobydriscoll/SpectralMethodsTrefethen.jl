# p23a.jl - eigenvalues of UNperturbed Laplacian on [-1,1]x[-1,1]
#         (compare p16.jl)

# Set up tensor product Laplacian and compute 4 eigenmodes:
N = 16
D,x = cheb(N)
y = x

D2 = D^2
D2 = D2[2:N,2:N]
L = -kron(I(N-1),D2) - kron(D2,I(N-1))                #  Laplacian
# f = [ exp(20*(y-x-1)) for y in y[2:N], x in x[2:N] ]
# L = L + Diagonal(vec(f))
λ,V = eigen(L)
λ,V = real(λ[1:4]),real(V[:,1:4])  # remove imaginary roundoff

# Reshape them to 2D grid, interpolate to finer grid, and plot:
xx = yy = -1:.02:1
U = zeros(N+1,N+1)
plt = plot(layout=(2,2))
for i = 1:4
    global U
    U[2:N,2:N] = reshape(V[:,i],N-1,N-1)
    s = Spline2D(reverse(y),reverse(x),reverse(reverse(U,dims=1),dims=2))
    UU = evalgrid(s,yy,xx)
    m,M = extrema(UU)
    scl = abs(m) > M ? m : M
    UU = UU/scl
    str = @sprintf("λ = %0.11f π^2/4",λ[i]/(π^2/4)) 
    contour!(xx,yy,UU,fill=true,subplot=i,levels=-1.1:.2:1.1,
      color=:balance,clims=(-1,1),
      xlim=(-1,1),ylim=(-1,1),aspect_ratio=1,title=str)
end
