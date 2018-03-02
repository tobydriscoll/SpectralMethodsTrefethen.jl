# p13.jl - solve linear BVP u_xx = exp(4x), u(-1)=u(1)=0

using SMiJ, Polynomials
N = 16;
(D,x) = cheb(N);
D2 = D^2;
D2 = D2[2:N,2:N];                   # boundary conditions
f = exp(4*x[2:N]);
u = D2\f;                           # Poisson eq. solved here
u = [0;u;0];
clf();
plot(x,u,".",markersize=12)
xx = -1:.01:1;
uu = polyval(polyfit(x,u),xx);      # interpolate grid data
plot(xx,uu);
grid(true);
exact = @. ( exp(4*xx) - sinh(4)*xx - cosh(4) )/16;
title("max err = $(norm(uu-exact,Inf))",fontsize=12);
