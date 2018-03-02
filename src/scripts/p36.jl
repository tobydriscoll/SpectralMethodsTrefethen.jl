# p36.m - Laplace eq. on [-1,1]x[-1,1] with nonzero BCs

using SMiJ, PyPlot, Interpolations
# Set up grid and 2D Laplacian, boundary points included:
N = 24; (D,x) = cheb(N); y = x;
xx = repmat(x',N+1,1); yy = repmat(y,1,N+1);
D2 = D^2; I = eye(N+1); L = kron(I,D2) + kron(D2,I);

# Impose boundary conditions by replacing appropriate rows of L:
b = @. (abs(xx[:])==1) | (abs(yy[:])==1);            # boundary pts
L[b,:] = 0; L[b,b] = eye(4*N);
rhs = zeros((N+1)^2);
rhs[b] = @. (yy[b]==1)*(xx[b]<0)*sin(pi*xx[b])^4 + .2*(xx[b]==1)*sin(3*pi*yy[b]);

# Solve Laplace equation, reshape to 2D, and plot:
u = L\rhs; uu = reshape(u,N+1,N+1);
xxx = yyy = -1:.04:1;
s = interpolate((x[end:-1:1],y[end:-1:1]),reduce(flipdim,uu,1:2),Gridded(Linear()));
clf();
surf(xxx,yyy,s[xxx,yyy]);
xlim(-1,1); ylim(-1,1); zlim(-.2,1);
gca()[:view_init](45,-110);
umid = uu[Int(N/2)+1,Int(N/2)+1];
text3D(0,.8,.4,"u(0,0) = $(signif(umid,9))");
