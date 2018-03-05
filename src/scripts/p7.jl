# p7.m - accuracy of periodic spectral differentiation

# Compute derivatives for various values of N:
Nmax = 50; allN = 6:2:Nmax; E = zeros(4,length(allN));
for j = eachindex(allN)
    N = allN[j]; h = 2*pi/N; x = h*(1:N);
    column = [0; @. .5*(-1)^(1:N-1)*cot((1:N-1)*h/2)];
    D = toeplitz(column,column[[1;N:-1:2]]);
    v = @. abs(sin(x))^3;                     # 3rd deriv in BV
    vprime = @. 3*sin(x)*cos(x)*abs(sin(x));
    E[1,j] = norm(D*v-vprime,Inf);
    v = @. exp(-sin(x/2)^(-2));               # C-infinity
    vprime = @. .5*v*sin(x)/sin(x/2)^4;
    E[2,j] = norm(D*v-vprime,Inf);
    v = @. 1/(1+sin(x/2)^2);                 # analytic in a strip
    vprime = @. -sin(x/2)*cos(x/2)*v^2;
    E[3,j] = norm(D*v-vprime,Inf);
    v = sin.(10*x); vprime = 10*cos.(10*x);   # band-limited
    E[4,j] = norm(D*v-vprime,Inf);
end

# Plot results:
clf();
titles = [L"|\sin(x)|^3",L"\exp(-\sin^{-2}(x/2))",L"1/(1+\sin^2(x/2))",L"\sin(10x)"];
for iplot = 1:4
    subplot(2,2,iplot);
    semilogy(allN,E[iplot,:],".-",markersize=10);
    axis([0,Nmax,1e-16,1e3]); grid(true);
    xticks(0:10:Nmax); yticks(10.0.^(-15:5:0));
    title(titles[iplot]);
    iplot > 2 ? xlabel("N") : nothing;
    iplot%2 > 0? ylabel("error") : nothing;
end
