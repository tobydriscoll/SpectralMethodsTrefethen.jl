# p11.jl - Chebyshev differentation of a smooth function

xx = -1:.01:1; uu = @. exp(xx)*sin(5*xx);
clf();
for N = [10 20]
    (D,x) = cheb(N); u = @. exp(x)*sin(5*x);
    subplot(2,2,1+2*(N==20));
    plot(x,u,".",markersize=10); grid(true);
    plot(xx,uu);
    title("u(x),  N=$N")
    error = D*u - @. exp(x)*(sin(5*x)+5*cos(5*x));
    subplot(2,2,2+2*(N==20));
    plot(x,error,".-",markersize=10); grid(true);
    title("error in u'(x),  N=$N");
end
