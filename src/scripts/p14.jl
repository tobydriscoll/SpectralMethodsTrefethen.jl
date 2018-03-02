# p14.jl - solve nonlinear BVP u_xx = exp(u), u(-1)=u(1)=0
#         (compare p13.jl)

using Polynomials
N = 16;
(D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
u = zeros(N-1);
change = 1; it = 0;
while change > 1e-15                   # fixed-point iteration
    unew = D2\exp.(u);
    change = norm(unew-u,Inf);
    u = unew; it = it+1;
end
u = [0;u;0];
clf();
plot(x,u,".",markersize=12)
xx = -1:.01:1;
uu = polyval(polyfit(x,u),xx);
plot(xx,uu), grid(true);
title("no. steps = $it      u(0) =$(u[Int(N/2+1)])")
