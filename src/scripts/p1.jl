# p1.jl - convergence of fourth-order finite differences

# For various N, set up grid in [-pi,pi] and function u(x):
Nvec = [ 2^j for j in 3:12 ]
plt = plot(xaxis=(:log,"N"),yaxis=(:log,"error"),
  title="Convergence of fourth-order finite differences" )
for N in Nvec
    h = 2π/N
    x = [ -π+j*h for j in 1:N ]
    u = @. exp(sin(x)^2)
    uprime = @. 2*sin(x)*cos(x)*u

    # Construct sparse fourth-order differentiation matrix:
    e = ones(N)
    D = sparse(1:N,[2:N;1],2e/3) - sparse(1:N,[3:N;1:2],e/12)
    D = (D-D')/h

    # Plot max(abs(D*u-uprime)):
    error = norm(D*u-uprime,Inf)
    scatter!([N],[error],m=:black)
end
plot!(Nvec,1.0./Nvec.^4,l=:dash)
annotate!(105,5e-8,text(L"N^{-4}"))
