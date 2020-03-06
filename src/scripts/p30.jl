# p30.jl - spectral integration, ODE style (compare p12.jl)

# Computation: various values of N, four functions:
Nmax = 50
E = zeros(4,Nmax)
for N = 1:Nmax
    i = 1:N
    D,x = cheb(N)
    x = x[i]
    w = inv(D[i,i])[1,:]
    f = @. abs(x)^3;     E[1,N] = abs(dot(w,f) - .5)
    f = @. exp(-x^(-2)); E[2,N] = abs(dot(w,f) - 2*(exp(-1)+sqrt(π)*(erf(1)-1)))
    f = @. 1/(1+x^2);    E[3,N] = abs(dot(w,f) - π/2)
    f = x.^10;           E[4,N] = abs(dot(w,f) - 2/11)
end

# Plot results:
labels = [L"|x|^3",L"e^{-x^2}",L"\frac{1}{1+x^2}",L"x^{10}"]
plt = plot(layout=(2,2),xaxis=((0,Nmax+1),0:10:Nmax),yaxis=(:log,"error",(1e-18,1e1),10.0.^(-15:5:0)))
for i = 1:4
    plot!(max.(1e-100,E[i,:]),m=4,subplot=i)
    annotate!(32,0.004,subplot=i,text(labels[i],:left))
end
