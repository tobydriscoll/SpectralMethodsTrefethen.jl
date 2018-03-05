# p27.jl - Solve KdV eq. u_t + uu_x + u_xxx = 0 on [-pi,pi] by
#         FFT with integrating factor v = exp(-ik^3t)*u-hat.

# Set up grid and two-soliton initial data:
N = 256; dt = .4/N^2; x = (2*pi/N)*(-N/2:N/2-1);
A = 25; B = 16; clf();
u = @. 3*A^2*sech(.5*(A*(x+2)))^2 + 3*B^2*sech(.5*(B*(x+1)))^2;
v = fft(u); k = [0:N/2-1;0;-N/2+1:-1]; ik3 = 1im*k.^3;

# Solve PDE and plot results:
tmax = 0.006; nplt = floor(Int,(tmax/25)/dt); nmax = round(Int,tmax/dt);
udata = u; tdata = [0.0];
for n = 1:nmax
    t = n*dt; g = -.5im*dt*k;
    E = exp.(dt*ik3/2); E2 = E.^2;
    a = g.*fft(real( ifft(     v    ) ).^2);
    b = g.*fft(real( ifft(E.*(v+a/2)) ).^2);     # 4th-order
    c = g.*fft(real( ifft(E.*v + b/2) ).^2);     # Runge-Kutta
    d = g.*fft(real( ifft(E2.*v+E.*c) ).^2);
    v = E2.*v + (E2.*a + 2*E.*(b+c) + d)/6;
    if mod(n,nplt) == 0
        u = real(ifft(v));
        udata = [udata u]; tdata = [tdata;t];
    end
end
mesh(x,tdata,udata',ccount=0,rcount=N);
gca()[:view_init](25,90-20);
xlabel("x"); ylabel("y"); grid(true);
xlim(-pi,pi); ylim(0,tmax); zlim(0,10000);|
zticks([0,2000]);
