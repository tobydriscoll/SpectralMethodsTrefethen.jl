# p21.jl - eigenvalues of Mathieu operator -u_xx + 2qcos(2x)u
#         (compare p8.jl and p. 724 of Abramowitz & Stegun)

N = 42
h = 2π/N
x = h*(1:N)
col = [ -0.5*(-1)^j/sin(h*j/2)^2 for j in 1:N-1 ]
D2 = toeplitz( [-π^2/(3*h^2)-1/6; col] )
qq = 0:.2:15
data = zeros(0,11)
for q in qq
    global data
    λ = eigvals( -D2 + 2q*Diagonal(cos.(2x)) )
    data = [ data; λ[1:11]' ]
end
plt = plot(qq,data[:,1:2:end],l=:darkblue,
  xaxis=("q",(0,15)),yaxis=("λ",(-24,32),-24:4:32) )
plot!(qq,data[:,2:2:end],l=(:dash,:darkblue))

