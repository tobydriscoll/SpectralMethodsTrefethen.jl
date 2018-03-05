# p2.jl - convergence of periodic spectral method (compare p1.jl)

clf();
# For various N (even), set up grid as before:
for N = 2:2:100
    h = 2*pi/N;  x = [ -pi + i*h for i = 1:N ];
    u = @. exp(sin(x)); uprime = @. cos(x)*u;

    # Construct spectral differentiation matrix:
    column = [ 0; @. .5*(-1)^(1:N-1)*cot((1:N-1)*h/2) ];
    D = toeplitz(column,column[[1;N:-1:2]]);

    # Plot max(abs(D*u-uprime)):
    error = norm(D*u-uprime,Inf);
    loglog(N,error,".",markersize=12);
end
title("Convergence of spectral differentiation")
xlabel("N");  ylabel("error");
