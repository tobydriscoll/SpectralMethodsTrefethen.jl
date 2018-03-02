# p2.jl - convergence of periodic spectral method (compare p1.jl)

using PyPlot
clf();
# For various N (even), set up grid as before:
for N = 2:2:100
    h = 2*pi/N;  x = [ -pi + i*h for i = 1:N ];
    u = @. exp(sin(x)); uprime = @. cos(x)*u;

    # Construct spectral differentiation matrix:
    D = [ 0.5*(-1)^(i-j)*cot((i-j)*h/2) for i=1:N, j=1:N ];
    D[1:N+1:end] = 0;

    # Plot max(abs(D*u-uprime)):
    error = norm(D*u-uprime,Inf);
    loglog(N,error,".",markersize=12);
end
title("Convergence of spectral differentiation")
xlabel("\$N\$");  ylabel("error");
