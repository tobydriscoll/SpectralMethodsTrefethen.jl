# p22.jl - 5th eigenvector of Airy equation u_xx = lambda*x*u

plt = plot(layout=(2,2))
for (k,N) in enumerate(12:12:48)
    D,x = cheb(N)
    D2 = D^2
    D2 = D2[2:N,2:N]
    λ,V = eigen(D2,Matrix(Diagonal(x[2:N])))      # generalized ev problem
    fifth = findall(λ.>0)[5]
    λ,v = λ[fifth],V[:,fifth]
    v = [0;v;0]                      # extend to booundary
    v = v/v[Int(N/2+1)]*airyai(0)    # normalize
    xx = -1:.01:1
    vv = polyval(polyfit(x,v),xx)
    plot!(xx,vv,subplot=k,
      title=@sprintf("N = %d     λ = %0.11f",N,λ) )
end  
