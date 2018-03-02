# p30b.jl - spectral integration, Clenshaw-Curtis style (compare p30.jl)

using SMiJ, PyPlot, SpecialFunctions, LaTeXStrings
# Computation: various values of N, four functions:
Nmax = 50; E = zeros(4,Nmax); clf();
for N = 1:Nmax
    (x,w) = clencurt(N);
    f = @. abs(x)^3;     E[1,N] = abs(dot(w,f) - .5);
    f = @. exp(-x^(-2)); E[2,N] = abs(dot(w,f) - 2*(exp(-1)+sqrt(pi)*(erf(1)-1)));
    f = @. 1/(1+x^2);    E[3,N] = abs(dot(w,f) - pi/2);
    f = x.^10;           E[4,N] = abs(dot(w,f) - 2/11);
end

# Plot results:
labels = [L"|x|^3",L"\exp(-x^2)",L"1/(1+x^2)",L"x^{10}"];
for iplot = 1:4
    subplot(2,2,iplot)
    semilogy(E[iplot,:]+1e-100,".-",markersize=10);
    axis([0,Nmax,1e-18,1e3]); grid(true);
    xticks(0:10:Nmax); yticks((10.0).^(-15:5:0));
    ylabel("error"); text(32,.004,labels[iplot]);
end
