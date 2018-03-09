# p23.jl - eigenvalues of perturbed Laplacian on [-1,1]x[-1,1]
#         (compare p16.jl)

# Set up tensor product Laplacian and compute 4 eigenmodes:
N = 16; (D,x) = cheb(N); y = x;
xx = x[2:N]; yy = y[2:N];
D2 = D^2; D2 = D2[2:N,2:N]; I = eye(N-1);
L = -kron(I,D2) - kron(D2,I);                #  Laplacian
f = @. exp(20*(yy-xx'-1));      # perturbation
L = L + diagm(f[:]);
(D,V) = eig(L); ii = sortperm(D)[1:4];
D = D[ii]; V = V[:,ii];

# Reshape them to 2D grid, interpolate to finer grid, and plot:
xx = yy = x[end:-1:1];
xxx = yyy = -1:.02:1;
uu = zeros(N+1,N+1);
(ay,ax) = (repmat([.56 .04],2,1),repmat([.1,.5],1,2)); clf();
for i = 1:4
    uu[2:N,2:N] = reshape(V[:,i],N-1,N-1);
    uu = uu/norm(uu[:],Inf);
    s = Spline2D(xx,yy,reduce(flipdim,uu,1:2));
    uuu = evalgrid(s,xxx,yyy);
    axes([ax[i],ay[i],.38,.38]);
    contour(xxx,yyy,uuu,levels=-.9:.2:.9);
    axis("square");
    title("eig = $(signif(D[i]/(pi^2/4),13)) π^2/4");
end
