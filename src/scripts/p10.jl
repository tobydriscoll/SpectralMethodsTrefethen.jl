# p10.jl - polynomials and corresponding equipotential curves

N = 16
plt = plot(layout=(2,2),grid=true)
for i = 1:2
    i==1 && ( (s,x) = ("equispaced points", [-1+2*n/N for n in 0:N]) );
    i==2 && ( (s,x) = ("Chebyshev points", [cos(n*π/N) for n in 0:N]) );
    p = poly(x);

    # Plot p(x) over [-1,1]:
    xx = -1:.005:1
    pp = p.(xx)
    scatter!(x,0*x,m=4,subplot=2i-1)
    plot!(xx,pp,subplot=2i-1,xaxis=(-1:0.5:1),title=s)

    # Plot equipotential curves:
    scatter!(real(x),imag(x),m=4,subplot=2i,xaxis=(-1.4,1.4),yaxis=(-1.12,1.12))
    xx,yy = -1.4:.02:1.4, -1.12:.02:1.12
    pp = [ p(x+1im*y) for y in yy, x in xx ]
    levels = 10.0.^(-4:0)
    contour!(xx,yy,abs.(pp),subplot=2i,levels=levels,title=s)
end
