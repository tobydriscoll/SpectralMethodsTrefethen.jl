# p9.jl - polynomial interpolation in equispaced and Chebyshev pts

using Polynomials
N = 16; xx = -1.01:.005:1.01;
using PyPlot;  clf();
for i = 1:2
    i==1 && ( (s,x) = ("equispaced points", -1 + 2*(0:N)/N) );
    i==2 && ( (s,x) = ("Chebyshev points", cos.(pi*(0:N)/N)) );
    subplot(1,2,i)
    u = 1./(1+16*x.^2);
    uu = 1./(1+16*xx.^2);
    p = polyfit(x,u);              # interpolation
    pp = p(xx);                    # evaluation of interpolant
    plot(x,u,".",markersize=10)
    plot(xx,pp)
    axis([-1.1,1.1,-1,1.5]); title(s);
    error = signif( norm(uu-pp,Inf), 5);
    text(-.5,-.5,"max error = $error",fontsize=8);
end
