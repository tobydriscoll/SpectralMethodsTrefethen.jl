# p16.jl - Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary

# Set up grids and tensor product Laplacian and solve for u:
N = 24; (D,x) = cheb(N); y = x;
xx = x[2:N]; yy = y[2:N];
f = 10*sin.(8*xx'.*(yy-1));
D2 = D^2; D2 = D2[2:N,2:N]; I = eye(N-1);
L = kron(I,D2) + kron(D2,I);                       # Laplacian
figure(1); clf(); spy(L);
tic(); u = L\f[:]; toc();          # solve problem and watch the clock

# Reshape long 1D results onto 2D grid (flipping orientation):
uu = zeros(N+1,N+1); uu[N:-1:2,N:-1:2] = reshape(u,N-1,N-1);
value = uu[Int(3N/4+1),Int(3N/4+1)];

# Interpolate to finer grid and plot:
xxx = yyy = -1:.04:1;
s = interpolate((x[end:-1:1],y[end:-1:1]),uu,Gridded(Linear()));
uuu = s[xxx,yyy];
figure(2); clf(); surf(xxx,yyy,uuu,rstride=1,cstride=1);
xlabel("x"); ylabel("y"); zlabel("u"); view(-37.5,30);
text3D(.4,-.3,-.3,"\$u(2^{-1/2},2^{-1/2})\$ = $(signif(value,11))",fontsize=9);
