# p2.jl - convergence of periodic spectral method (compare p1.jl)

# For various N (even), set up grid as before:
plt = plot(xaxis=(:log,"N"),yaxis=(:log,"error"),
  title="Convergence of spectral differentiation" )
for N in 2:2:100
    h = 2π/N
    x = [ -π + j*h for j in 1:N ]
    u = @. exp(sin(x))
    uprime = @. cos(x)*u

    # Construct spectral differentiation matrix:
    column = [ 0; @. .5*(-1)^(1:N-1)*cot((1:N-1)*h/2) ]
    D = toeplitz(column,column[[1;N:-1:2]])

    # Plot max(abs(D*u-uprime)):
    error = norm(D*u-uprime,Inf)
    scatter!([N],[error],m=:black)
end
