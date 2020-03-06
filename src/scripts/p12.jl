# p12.jl - accuracy of Chebyshev spectral differentiation
#         (compare p7.jl)

# Compute derivatives for various values of N:
Nmax = 50
E = zeros(4,Nmax)
# 3rd deriv in BV, C^∞, analytic, polynomial
funs = [ x->abs(x)^3, x-> exp(-x^(-2)), x->1/(1+x^2), x->x^10 ]
derivs = [ x->3*x*abs(x) , x->2exp(-x^(-2))/x^3, x->-2x/(1+x^2)^2, x->10x^9 ]
for N in 1:Nmax
    D,x = cheb(N)
    for i in eachindex(funs)
        v = funs[i].(x)
        vprime = derivs[i].(x)
        E[i,N] = norm(D*v-vprime,Inf)
    end
end

# Plot results:
titles = [L"|x|^3",L"\exp(-x^2)",L"1/(1+x^2)",L"x^{10}"]
plt = plot(layout=(2,2),grid=true,xaxis=((0,Nmax+2),0:10:Nmax), yaxis=(:log,(1e-16,1e3),10.0.^(-15:5:0)))
for iplot = 1:4
    plot!(1:Nmax,E[iplot,:],m=4,subplot=iplot,title=titles[iplot])
    iplot > 2 && xlabel!("N",subplot=iplot)
    iplot%2 > 0 && ylabel!("error",subplot=iplot)
end
