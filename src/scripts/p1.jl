# p1.jl - convergence of fourth-order finite differences
using PyPlot

# For various N, set up grid in [-pi,pi] and function u(x):
Nvec = 2.^(3:12);  clf()
for N = Nvec
    h = 2*pi/N; x = -pi + (1:N)*h;
    u = @. exp(sin(x)^2);
    uprime = @. 2*sin(x)*cos(x)*u;

    # Construct sparse fourth-order differentiation matrix:
    e = ones(N);
    D = sparse(1:N,[2:N;1],2*e/3) - sparse(1:N,[3:N;1:2],e/12);
    D = (D-D')/h;

    # Plot max(abs(D*u-uprime)):
    error = norm(D*u-uprime,Inf);
    loglog(N,error,".",markersize=12);
end
title("Convergence of fourth-order finite differences");
xlabel("\$N\$");  ylabel("error");
loglog(Nvec,1./float(Nvec).^4,"--");
text(105,5e-8,"\$N^{-4}\$",fontsize=18);
