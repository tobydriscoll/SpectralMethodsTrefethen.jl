# p27.jl - Solve KdV eq. u_t + uu_x + u_xxx = 0 on [-pi,pi] by
#         FFT with integrating factor v = exp(-ik^3t)*u-hat.

# Set up grid and two-soliton initial data:
N = 256
dt = .4/N^2
x = (2π/N)*(-N/2:N/2-1)
A,B = 25,16
u = @. 3A^2*sech(0.5*(A*(x+2)))^2 + 3B^2*sech(0.5*(B*(x+1)))^2
û = fft(u)
k = [0:N/2-1;0;-N/2+1:-1]
ik3 = 1im*k.^3
g = -0.5im*dt*k

# Solve PDE and plot results:
tmax = 0.006
nplt = floor(Int,(tmax/25)/dt)
nmax = round(Int,tmax/dt)
udata = u
tdata = [0.0]
for n = 1:nmax
    global û
    t = n*dt
    E = exp.(dt*ik3/2)
    E2 = E.^2;
    a = g.*fft(real( ifft(     û    ) ).^2)
    b = g.*fft(real( ifft(E.*(û+a/2)) ).^2)     # 4th-order
    c = g.*fft(real( ifft(E.*û + b/2) ).^2)     # Runge-Kutta
    d = g.*fft(real( ifft(E2.*û+E.*c) ).^2)
    û = @. E2*û + (E2*a + 2*E*(b+c) + d)/6
    global udata
    global tdata
    if mod(n,nplt) == 0
        u = real(ifft(û))
        udata = [udata u]
        tdata = [tdata;t]
    end
end
plt = surface(x,tdata,udata',camera=(-20,55),
  xaxis=((-π,π),"x"),yaxis=("y"),zaxis=((0,2200),[0,0,2000]) )
