# p8.jl - eigenvalues of harmonic oscillator -u"+x^2 u on R

L = 8                             # domain is [-L L], periodic
for N in 6:6:36
    h = 2π/N
    x = [ j*h for j in 1:N ]
    x = L*(x.-π)/π
    column = [ -π^2/(3*h^2)-1/6; [-0.5*(-1)^j/sin(h*j/2)^2 for j in 1:N-1] ]
    D2 = (π/L)^2*toeplitz(column)  # 2nd-order differentiation
    λ = eigvals(-D2 + Diagonal(x.^2))
    @show N
    [ println(λ[i]) for i=1:4 ]
    println("")
end
plt = nothing
