# p38.jl - solve u_xxxx = exp(x), u(-1)=u(1)=u'(-1)=u'(1)=0
#         (compare p13.jl)

using PyPlot, Polynomials
# Construct discrete biharmonic operator:
N = 15; (D,x) = cheb(N);
S = diagm([0; 1 ./(1-x[2:N].^2); 0]);
D4 = (diagm(1-x.^2)*D^4 - 8*diagm(x)*D^3 - 12*D^2)*S;
D4 = D4[2:N,2:N];

# Solve boundary-value problem and plot result:
f = @. exp(x[2:N]); u = D4\f; u = [0;u;0];
clf(); axes([.1,.4,.8,.5]);
plot(x,u,".",markersize=10);
axis([-1,1,-.01,.06]); grid(true);
xx = (-1:.01:1);
uu = (1-xx.^2).*polyval(polyfit(x,S*u),xx);
plot(xx,uu);

# Determine exact solution and print maximum error:
A = [1 -1 1 -1; 0 1 -2 3; 1 1 1 1; 0 1 2 3];
V = xx.^(0:3)';
c = A\exp.([-1,-1,1,1]); exact = exp.(xx) - V*c;
title("max err = $(signif(norm(uu-exact,Inf),5))",fontsize=12);
