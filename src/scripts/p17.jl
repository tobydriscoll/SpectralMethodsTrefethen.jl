# p17.jl - Helmholtz eq. u_xx + u_yy + (k^2)u = f
#         on [-1,1]x[-1,1]    (compare p16.jl)

# Set up spectral grid and tensor product Helmholtz operator:
N = 24; (D,x) = cheb(N); y = x;
xx = x[2:N]; yy = y[2:N];
f = @. exp(-10*((yy-1)^2+(xx'-.5)^2));
D2 = D^2; D2 = D2[2:N,2:N]; I = eye(N-1);
k = 9;
L = kron(I,D2) + kron(D2,I) + k^2*eye((N-1)^2);

# Solve for u, reshape to 2D grid, and plot:
u = L\f[:];
uu = zeros(N+1,N+1); uu[N:-1:2,N:-1:2] = reshape(u,N-1,N-1);
xxx = yyy = -1:.0333:1;
s = Spline2D(x[end:-1:1],y[end:-1:1],uu);
uuu = evalgrid(s,xxx,yyy);
figure(1); clf(); surf(xxx,yyy,uuu,rstride=1,cstride=1);
xlabel("x"); ylabel("y"); zlabel("u"); view(-37.5,30);
value = signif(uu[Int(N/2+1),Int(N/2+1)],10);
text3D(.2,1,.022,"u(0,0) = $value");
figure(2); clf(); contour(xxx,yyy,uuu,10); axis("square");
