# p28b.jl - eigenmodes of Laplacian on the disk

using SMiJ,PyPlot
# r coordinate, ranging from -1 to 1 (N must be odd)
N = 25; N2 = Int((N-1)/2);
(D,r) = cheb(N); D2 = D^2;
D1 = D2[2:N2+1,2:N2+1]; D2 = D2[2:N2+1,N:-1:N2+2];
E1 =  D[2:N2+1,2:N2+1]; E2 =  D[2:N2+1,N:-1:N2+2];

# t = theta coordinate, ranging from 0 to 2*pi (M must be even):
M = 20; dt = 2*pi/M; t = dt*(1:M); M2 = Int(M/2);
D2t = toeplitz([-pi^2/(3*dt^2)-1/6; @. .5*(-1)^(2:M)/sin(dt*(1:M-1)/2)^2]);

# Laplacian in polar coordinates:
R = diagm(1./r[2:N2+1]);
Z = zeros(M2,M2); I = eye(M2);
L = kron(D1+R*E1,eye(M)) + kron(D2+R*E2,[Z I;I Z]) + kron(R^2,D2t);

# Compute 25 eigenmodes:
index = 1:25;
(Lam,V) = eig(-L); ii = sortperm(abs.(Lam))[index];
Lam = Lam[ii]; V = V[:,ii];
Lam = sqrt.(real(Lam/Lam[1]));

# Plot nodal lines:
(rr,tt) = (r[1:N2+1],[0;t]);
(xx,yy) = @. (cos(tt)*rr',sin(tt)*rr');
z = exp.(1im*pi*(-100:100)/100);  clf();
for i = 1:25
    subplot(5,5,i);
    u = reshape(real(V[:,i]),M,N2);
    u = [zeros(M+1) u[[M;1:M],:]];
    u = u/norm(u[:],Inf);
    plot(real(z),imag(z));
    xlim(-1.07,1.07); ylim(-1.07,1.07); axis("off"); axis("equal");
    contour(xx,yy,u,levels=[0]);
    title("$(signif(Lam[i],5))",fontsize=8);
end
