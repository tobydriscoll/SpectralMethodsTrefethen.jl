# p18.jl - Chebyshev differentiation via FFT (compare p11.jl)

xx = -1:.01:1; ff = @. exp(xx)*sin(5*xx); clf();
for N = [10 20]
    x = cheb(N)[2]; f = @. exp(x)*sin(5*x);
    axes([.15, .66-.4*(N==20), .31, .28]);
    plot(x,f,"k.",markersize=6); grid(true);
    plot(xx,ff);
    title("f(x), N=$N");
    error = chebfft(f) - @. exp(x)*(sin(5*x)+5*cos(5*x));
    axes([.55, .66-.4*(N==20), .31, .28]);
    plot(x,error,".-",markersize=10); grid(true);
    title("error in f'(x),  N=$N");
end
