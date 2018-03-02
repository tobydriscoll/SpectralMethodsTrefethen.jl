# p18.jl - Chebyshev differentiation via FFT (compare p11.jl)

using SMiJ, PyPlot
xx = -1:.01:1; ff = @. exp(xx)*sin(5*xx); clf();
for N = [10 20]
    x = cheb(N)[2]; f = @. exp(x)*sin(5*x);
    subplot(2,2,1+2*(N==20));
    plot(x,f,".",markersize=10); grid(true);
    plot(xx,ff);
    title("f(x), N=$N");
    error = chebfft(f) - @. exp(x)*(sin(5*x)+5*cos(5*x));
    subplot(2,2,2+2*(N==20));
    plot(x,error,".-",markersize=10); grid(true);
    title("error in f'(x),  N=$N");
end
