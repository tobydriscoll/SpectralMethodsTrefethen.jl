# p10.jl - polynomials and corresponding equipotential curves
using PyPlot;  clf();
using Polynomials
N = 16;
xx = -1.01:.005:1.01;
for i = 1:2
    i==1 && ( (s,x) = ("equispaced points", -1 + 2*(0:N)/N) );
    i==2 && ( (s,x) = ("Chebyshev points", cos.(pi*(0:N)/N)) );
    p = poly(x);

    # Plot p(x) over [-1,1]:
    xx = -1:.005:1; pp = p(xx);
    subplot(2,2,2*i-1);
    plot(x,0*x,"k.",markersize=10);
    plot(xx,pp);  grid(true);
    xticks(-1:.5:1);  title(s);

    # Plot equipotential curves:
    subplot(2,2,2*i)
    plot(real(x),imag(x),"k.",markersize=10);
    axis([-1.4,1.4,-1.12,1.12]);
    xx = -1.4:.02:1.4; yy = -1.12:.02:1.12;
    zz = xx' .+ 1im*yy;
    pp = p(zz); levels = 10.0.^(-4:0);
    contour(xx,yy,abs.(pp),levels); title(s);
end
