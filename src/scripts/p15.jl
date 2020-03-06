# p15.jl - solve eigenvalue BVP u_xx = lambda*u, u(-1)=u(1)=0

N = 36
D,x = cheb(N)
D2 = D^2
D2 = D2[2:N,2:N]
lam,V = eigen(-D2)
plt = plot(layout=(6,1),xaxis=false,yaxis=false,grid=false)
for (sp,j) in enumerate(5:5:30)        # plot 6 eigenvectors
    u = [ 0; V[:,j]; 0 ]
    scatter!(x,u,m=4,subplot=sp)
    xx = -1:.01:1
    uu = chebinterp(u).(xx)
    plot!(xx,uu,subplot=sp)
    annotate!(-0.4,-0.25,text("eig $j = $(-lam[j]*4/π^2) π^2/4",7,:left,:top),subplot=sp)
    annotate!(0.7,-0.25,text("$(round(4*N/(pi*j),sigdigits=2))  ppw",7,:left,:top),subplot=sp)
end
