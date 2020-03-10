# p20.jl - 2nd-order wave eq. in 2D via FFT (compare p19)

# Grid and initial data:
N = 24
x = y = cheb(N)[2]
dt = 6/N^2
xr,yr = reverse(x),reverse(y)
plotgap = round(Int,(1/3)/dt)
dt = (1/3)/plotgap;
V = [ exp(-40*((x-0.4)^2 + y^2)) for y in y, x in x ]
Vold = V
iξ = 1im*[0:N-1;0;1-N:-1]  # first derivative in wavenumber space
iξsq = -[0:N;1-N:-1].^2    # second derivative

# Time-stepping by leap frog formula:
plt = plot(layout=(2,2))
for n in 0:3plotgap
    global V
    global Vold
    t = n*dt
    if mod(n,plotgap)==0     # plots at multiples of t=1/3
        i = Int(n/plotgap+1)
        xx = yy = -1:1/16:1;
        VV = hcat([ chebinterp(V[:,j]).(yy) for j in 1:N+1 ]...)
        VV = vcat([ chebinterp(VV[i,:]).(xx)' for i in 1:length(yy) ]...)
        str = @sprintf("t = %0.4f",t)
        surface!(xx,yy,VV,subplot=i,clims=(-0.2,1.0),color=:viridis,cam=(-37.5,30),
          xlim=(-1,1),ylim=(-1,1),zlim=(-0.15,1),title=str)
    end
    Vxx = zeros(N+1,N+1)
    Vyy = zeros(N+1,N+1)
    ii = 2:N
    for i in 2:N                # 2nd derivs wrt x in each row
        v = V[i,:]
        u = [v;reverse(v[ii])]
        û = real(fft(u))
        W1 = real(ifft(iξ.*û))        # diff wrt theta
        W2 = real(ifft(iξsq.*û))      # diff^2 wrt theta
        Vxx[i,ii] = @. W2[ii]/(1-x[ii]^2) - x[ii]*W1[ii]/(1-x[ii]^2)^(3/2)
    end
    for j in 2:N                # 2nd derivs wrt y in each column
        v = V[:,j]
        u = [v;reverse(v[ii])]
        û = real(fft(u))
        W1 = real(ifft(iξ.*û))        # diff wrt theta
        W2 = real(ifft(iξsq.*û))      # diff^2 wrt theta
        Vyy[ii,j] = @. W2[ii]/(1-y[ii]^2) - y[ii]*W1[ii]/(1-y[ii]^2)^(3/2)
    end
    Vnew = 2V - Vold + dt^2*(Vxx+Vyy)
    Vold,V = V,Vnew
end
