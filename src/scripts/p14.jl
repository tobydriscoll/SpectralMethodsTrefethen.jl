# p14.jl - solve nonlinear BVP u_xx = exp(u), u(-1)=u(1)=0
#         (compare p13.jl)

N = 16
D,x = cheb(N)
D2 = D^2
D2 = D2[2:N,2:N]
u = zeros(N-1)
it = 0
while true                # fixed-point iteration
    global u
    global it
    unew = D2\exp.(u)
    change = norm(unew-u,Inf)
    it += 1
    u = unew
    (change < 1e-15 || it > 99) && break
end
u = [0;u;0]
plt = scatter(x,u,grid=true)
xx = -1:.01:1
uu = polyval(polyfit(x,u),xx)
plot!(xx,uu,title="no. steps = $it,    u(0) =$(u[Int(N/2+1)])")
