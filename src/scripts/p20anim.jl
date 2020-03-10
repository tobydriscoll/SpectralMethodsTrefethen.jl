# p20jl - 2nd-order wave eq. in 2D via FFT, w/animation

# Grid and initial data:
N = 24
x = y = cheb(N)[2]
dt = 6/N^2
Tf = 1.8
dt = Tf/round(Int,Tf/dt)
xr,yr = reverse(x),reverse(y)
V = [ exp(-40*((x-0.4)^2 + y^2)) for y in y, x in x ]
Vold = V
iξ = 1im*[0:N-1;0;1-N:-1]  # first derivative in wavenumber space
iξsq = -[0:N;1-N:-1].^2    # second derivative
xx = yy = -1:1/60:1       # plotting grid
Vxx = zeros(N+1,N+1)
Vyy = zeros(N+1,N+1)
ii = 2:N

# Time-stepping by leap frog formula:
anim = @animate for n in 0:Int(Tf/dt)
    global V
    global Vold

    VV = hcat([ chebinterp(V[:,j]).(yy) for j in 1:N+1 ]...)
    VV = vcat([ chebinterp(VV[i,:]).(xx)' for i in 1:length(yy) ]...)
    str = @sprintf("t = %0.4f",n*dt) 
    heatmap(xx,yy,VV,clims=(-0.8,0.8),color=:balance,aspect_ratio=1,
         xlim=(-1,1),ylim=(-1,1),title=str)

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

end every 2
plt = gif(anim,"p20anim.gif",fps=15)