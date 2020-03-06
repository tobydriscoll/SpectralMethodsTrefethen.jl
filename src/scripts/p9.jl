# p9.jl - polynomial interpolation in equispaced and Chebyshev pts

N = 16
xx = -1.01:.005:1.01
plt = plot(layout=(2,1))
for i = 1:2
    i==1 && ( (s,x) = ("equispaced points", [-1+2*n/N for n in 0:N]) )
    i==2 && ( (s,x) = ("Chebyshev points", [cos(n*π/N) for n in 0:N]) )
    u = @. 1/(1+16*x^2)
    uu = @. 1/(1+16*xx^2)
    p = chebinterp(u,x)             # interpolation
    pp = p.(xx)                     # evaluation of interpolant
    scatter!(x,u,m=4,subplot=i)
    plot!(xx,pp,subplot=i,xlim=(-1.1,1.1),ylim=(-1,1.5),title=s)
    error = round(norm(uu-pp,Inf),sigdigits=5)
    annotate!(-0.5,-0.5,text("max error = $error",8),subplot=i)
end
