# p7.jl - accuracy of periodic spectral differentiation

# Compute derivatives for various values of N:
Nmax = 50
allN = 6:2:Nmax
E = zeros(4,length(allN))
for N in allN
    h = 2π/N
    x = [ j*h for j in 1:N ]
    column = [0; [ 0.5*(-1)^j*cot(j*h/2) for j in 1:N-1 ] ]
    D = toeplitz(column,column[[1;N:-1:2]])
    v = @. abs(sin(x))^3                     # 3rd deriv in BV
    vprime = @. 3*sin(x)*cos(x)*abs(sin(x))
    j = round(Int,N/2-2)
    E[1,j] = norm(D*v-vprime,Inf)
    v = @. exp(-sin(x/2)^(-2))               # C-infinity
    vprime = @. .5*v*sin(x)/sin(x/2)^4
    E[2,j] = norm(D*v-vprime,Inf)
    v = @. 1/(1+sin(x/2)^2)                  # analytic in a strip
    vprime = @. -sin(x/2)*cos(x/2)*v^2
    E[3,j] = norm(D*v-vprime,Inf)
    v = sin.(10*x); vprime = 10*cos.(10*x)   # band-limited
    E[4,j] = norm(D*v-vprime,Inf)
end

# Plot results:
titles = [L"|\sin(x)|^3",L"\exp(-\sin^{-2}(x/2))",L"1/(1+\sin^2(x/2))",L"\sin(10x)"]
plt = plot(layout=(2,2),
    xaxis=((0,Nmax+2),0:10:Nmax),yaxis=(:log,(1e-16,1e3),10.0.^(-15:5:0)))
for iplot = 1:4
    plot!(allN,E[iplot,:],m=4,subplot=iplot,title=titles[iplot])
    iplot > 2 && xlabel!("N",subplot=iplot)
    iplot%2 > 0 && ylabel!("error",subplot=iplot)
end
