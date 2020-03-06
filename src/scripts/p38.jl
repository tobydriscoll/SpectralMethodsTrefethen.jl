# p38.jl - solve u_xxxx = exp(x), u(-1)=u(1)=u'(-1)=u'(1)=0
#         (compare p13.jl)

# Construct discrete biharmonic operator:
N = 15
D,x = cheb(N)
s = [ 1/(1-x^2) for x in x[2:N] ]
S = Diagonal([0;s;0])
D4 = (Diagonal(1 .- x.^2)*D^4 - 8*Diagonal(x)*D^3 - 12*D^2)*S
D4 = D4[2:N,2:N]

# Solve boundary-value problem and plot result:
f = [ exp(x) for x in x[2:N] ]
u = D4\f
u = [0;u;0]
plt = scatter(x,u)
#axis([-1,1,-.01,.06]); grid(true);
xx = -1:.01:1
uu = (1 .- xx.^2).*polyval(polyfit(x,S*u),xx);
plot!(xx,uu)

# Determine exact solution and print maximum error:
c = [1 -1 1 -1; 0 1 -2 3; 1 1 1 1; 0 1 2 3] \ exp.([-1,-1,1,1])
exact = exp.(xx) - sum( c[i]*xx.^(i-1) for i in 1:4 )
title!( @sprintf("max err = %.5g",norm(uu-exact,Inf)) )
