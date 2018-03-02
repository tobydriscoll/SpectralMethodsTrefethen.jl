# p19.jl - 2nd-order wave eq. on Chebyshev grid (compare p6.jl)

using PyPlot
# Time-stepping by leap frog formula:
N = 80; x = cheb(N)[2]; dt = 8/N^2;
v = @. exp(-200*x^2); vold = @. exp(-200*(x-dt)^2);
tmax = 4; tplot = .075;
plotgap = round(Int,tplot/dt); dt = tplot/plotgap;
nplots = round(Int,tmax/tplot);
plotdata = [v  zeros(N+1,nplots)]; tdata = 0;
for i = 1:nplots
    for n = 1:plotgap
        w = chebfft(chebfft(v)); w[1] = 0; w[N+1] = 0;
        vnew = 2*v - vold + dt^2*w; vold = v; v = vnew;
    end
    plotdata[:,i+1] = v; tdata=[tdata;dt*i*plotgap];
end

# Plot results:
clf(); mesh(x,tdata,plotdata',ccount=0,rcount=N+1);
axis([-1,1,0,tmax]); zlim(-2,2); grid(false);
gca()[:view_init](70,10+90);
xlabel("x"); ylabel("t"); zlabel("u");
