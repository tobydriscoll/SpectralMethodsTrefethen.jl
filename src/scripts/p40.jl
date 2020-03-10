# p40.jl - eigenvalues of Orr-Sommerfeld operator (compare p38)

R = 5772
plt = plot(layout=(2,2),aspect_ratio=1,xlim=(-0.8,0.2),ylim=(-1,0))
for (i,N) in enumerate(40:20:100)
    # 2nd- and 4th-order differentiation matrices:
    D,x = cheb(N)
    D2 = D^2
    D2 = D2[2:N,2:N]
    S = Diagonal( [0; [1/(1-x^2) for x in x[2:N]]; 0] )
    D4 = (Diagonal(1 .- x.^2)*D^4 - 8*Diagonal(x)*D^3 - 12*D^2)*S
    D4 = D4[2:N,2:N]

    # Orr-Sommerfeld operators A,B and generalized eigenvalues:
    A = (D4-2*D2+I)/R - 2im*I - 1im*Diagonal([1-x^2 for x in x[2:N]])*(D2-I)
    B = D2-I
    λ = eigvals(A,B)
    scatter!(real(λ),imag(λ),subplot=i,
      title=@sprintf("N = %d,   \$\\lambda_{max}\$ = %.7g",N,maximum(real(λ))))
end
