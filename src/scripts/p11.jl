# p11.jl - Chebyshev differentation of a smooth function

xx = -1:.01:1
uu = @. exp(xx)*sin(5*xx)
plt = plot(layout=(2,2),grid=true)
for (i,N) in enumerate([10,20])
    D,x = cheb(N)
    u = @. exp(x)*sin(5*x)

    scatter!(x,u,m=4,subplot=2i-1)
    plot!(xx,uu,subplot=2i-1,title="u(x),  N=$N")

    error = D*u - @. exp(x)*(sin(5*x)+5*cos(5*x))
    plot!(x,error,m=4,subplot=2i,title="error in u'(x),  N=$N")
end
