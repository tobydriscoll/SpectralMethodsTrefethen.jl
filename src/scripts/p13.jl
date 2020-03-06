# p13.jl - solve linear BVP u_xx = exp(4x), u(-1)=u(1)=0

N = 16
D,x = cheb(N)
D2 = D^2
D2 = D2[2:N,2:N]                    # boundary conditions
f = @. exp(4*x[2:N])
u = D2\f                            # Poisson eq. solved here
u = [0;u;0]
plt = scatter(x,u,m=4,grid=true)
xx = -1:.01:1;
uu = chebinterp(u).(xx)             # interpolate grid data
plot!(xx,uu)
exact = @. ( exp(4*xx) - sinh(4)*xx - cosh(4) )/16;
title!("max err = $(round(norm(uu-exact,Inf),sigdigits=4))")
