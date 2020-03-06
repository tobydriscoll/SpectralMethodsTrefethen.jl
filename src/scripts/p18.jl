# p18.jl - Chebyshev differentiation via FFT (compare p11.jl)

xx = -1:.01:1
ff = @. exp(xx)*sin(5*xx)
plt = plot(layout=(2,2),grid=true)
for (i,N) in enumerate([10,20])
    D,x = cheb(N)
    f = @. exp(x)*sin(5*x)

    scatter!(x,f,m=4,subplot=2i-1)
    plot!(xx,ff,subplot=2i-1,title="f(x),  N=$N")

    error = chebfft(f) - @. exp(x)*(sin(5*x)+5*cos(5*x))
    plot!(x,error,m=4,subplot=2i,title="error in f'(x),  N=$N")
end
