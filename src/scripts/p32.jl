# p32.jl - solve u_xx = exp(4x), u(-1)=0, u(1)=1 (compare p13.jl)

using Polynomials
N = 16;
(D,x) = cheb(N);
D2 = D^2;
D2 = D2[2:N,2:N];                   # boundary conditions
f = exp(4*x[2:N]);
u = D2\f;                           # Poisson eq. solved here
u = [0;u;0] + (x+1)/2;
clf();
plot(x,u,".",markersize=10);
xx = -1:.01:1;
uu = polyval(polyfit(x,u),xx);
plot(xx,uu); grid(true);
exact = @. (exp(4*xx) - sinh(4)*xx - cosh(4))/16 + (xx+1)/2;
title("max err = $(signif(norm(uu-exact,Inf),4))",fontsize=12);
