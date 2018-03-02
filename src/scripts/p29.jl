# p29.jl - solve Poisson equation on the unit disk
#         (compare p16.jl and p28.jl)

using SMiJ,PyPlot
# Laplacian in polar coordinates:
N = 25; N2 = Int((N-1)/2);
(D,r) = cheb(N); D2 = D^2;
D1 = D2[2:N2+1,2:N2+1]; D2 = D2[2:N2+1,N:-1:N2+2];
E1 =  D[2:N2+1,2:N2+1]; E2 =  D[2:N2+1,N:-1:N2+2];
M = 20; dt = 2*pi/M; t = dt*(1:M); M2 = Int(M/2);
D2t = [@. .5*(-1)^(i-j+1)/sin(dt*(i-j)/2)^2 for i=1:M, j=1:M];
D2t[1:M+1:end] = -pi^2/(3*dt^2)-1/6;
R = diagm(1./r[2:N2+1]); Z = zeros(M2,M2); I = eye(M2);
L = kron(D1+R*E1,eye(M)) + kron(D2+R*E2,[Z I;I Z]) + kron(R^2,D2t);

# Right-hand side and solution for u:
(rr,tt) = (r[2:N2+1]',t);
f = @. -rr^2*sin(tt/2)^4 + sin(6*tt)*cos(tt/2)^2; u = L\f[:];

# Reshape results onto 2D grid and plot them:
u = reshape(u,M,N2); u = [zeros(M+1) u[[M;1:M],:]];
(rr,tt) = (r[1:N2+1],t[[M;1:M]]);
(xx,yy) = @. (cos(tt)*rr',sin(tt)*rr');
clf();
surf(xx,yy,u);
gca()[:view_init](40,-70);
xlim(-1,1); ylim(-1,1); zlim(-.01,.05);
xlabel("x"); ylabel("y"); zlabel("z");
