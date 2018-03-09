# p12.jl - accuracy of Chebyshev spectral differentiation
#         (compare p7.jl)

# Compute derivatives for various values of N:
Nmax = 50; allN = 6:2:Nmax; E = zeros(4,Nmax);
for N = 1:Nmax
    (D,x) = cheb(N);
    v = @. abs(x)^3; vprime = @. 3*x*abs(x);   # 3rd deriv in BV
    E[1,N] = norm(D*v-vprime,Inf);
    v = @. exp(-x^(-2)); vprime = @. 2*v/x^3;  # C-infinity
    E[2,N] = norm(D*v-vprime,Inf);
    v = @. 1/(1+x^2); vprime = @. -2*x*v^2;    # analytic in [-1,1]
    E[3,N] = norm(D*v-vprime,Inf);
    v = x.^10; vprime = 10*x.^9;               # polynomial
    E[4,N] = norm(D*v-vprime,Inf);
end

# Plot results:
titles = [L"|x|^3",L"\exp(-x^2)",L"1/(1+x^2)",L"x^{10}"];  clf();
for iplot = 1:4
    subplot(2,2,iplot);
    semilogy(1:Nmax,E[iplot,:],".-",markersize=6);
    axis([0,Nmax,1e-16,1e3]); grid(true);
    xticks(0:10:Nmax); yticks(10.0.^(-15:5:0));
    title(titles[iplot]);
    iplot > 2 ? xlabel("N") : nothing;
    iplot%2 > 0? ylabel("error") : nothing;
end
