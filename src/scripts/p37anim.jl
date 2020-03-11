# p37.jl - 2D "wave tank" with Neumann BCs for |y|=1

# x variable in [-A,A], Fourier:
A,Nx = 3,50
dx = 2A/Nx
x = -A .+ dx*(1:Nx)
col = [ 0.5*(-1).^(j+1)/sin((π*dx/A)*j/2)^2 for j in 1:Nx-1 ]
D2x = (π/A)^2*toeplitz([-1/(3*(dx/A)^2)-1/6; col])

# y variable in [-1,1], Chebyshev:
Ny = 15
Dy,y = cheb(Ny)
D2y = Dy^2
BC = -Dy[[1,Ny+1],[1,Ny+1]] \ Dy[[1,Ny+1],2:Ny]

# Initial data:
V = [ exp(-8*((x+1.5)^2+y^2)) for y in y, x in x ]
dt = 5/(Nx+Ny^2)
Vold = [ exp(-8*((x+dt+1.5)^2+y^2)) for y in y, x in x ]

# Time-stepping by leap frog formula:
plotgap = round(Int,0.05/dt)
dt = 0.05/plotgap
xx,yy = (-30A:30A)/30,(-40:40)/40
anim = @animate for n = 0:4/dt
    global V
    global Vold
    t = n*dt
    #s = Spline2D(reverse(y),x,reverse(V,dims=1))
    #VV = evalgrid(s,yy,xx)
    VV = hcat([ chebinterp(V[:,j]).(yy) for j in 1:Nx ]...)
    VV = vcat([ fourinterp(VV[i,:]).(π*(xx/A.+1))' for i in 1:length(yy) ]...)
    heatmap(xx,yy,VV,color=:balance,clims=(-1,1),
            xaxis=-3:3,yaxis=-1:1,aspect_ratio=1,
            title=@sprintf("t = %0.2f",t) )
    Vnew = 2*V - Vold + dt^2*(V*D2x +D2y*V)
    Vold,V = V,Vnew
    V[[1,Ny+1],:] = BC*V[2:Ny,:]       # Neumann BCs for |y|=1
end every plotgap
plt = gif(anim,"p37anim.gif",fps=15)