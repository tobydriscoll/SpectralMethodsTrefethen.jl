# p6u.jl - variable coefficient wave equation - UNSTABLE VARIANT

using PyPlot
# Grid, variable coefficient, and initial data:
N = 128; h = 2*pi/N; x = h*(1:N);
c = @. .2 + sin(x-1)^2;
t = 0; dt = 1.9/N;
v = @. exp(-100*(x-1)^2);
vold = @. exp(-100*(x-.2*dt-1)^2);

# Time-stepping by leap frog formula:
tmax = 8; tplot = .15; clf();
plotgap = round(Int,tplot/dt);
nplots = round(Int,tmax/tplot);
data = [v zeros(N,nplots)]; tdata = t;
for i = 1:nplots
    for n = 1:plotgap
        t = t+dt;
        v_hat = fft(v);
        w_hat = 1im*[0:N/2-1; 0; -N/2+1:-1] .* v_hat;
        w = real.(ifft(w_hat));
        vnew = vold - 2*dt*c.*w;        # leap frog formula
        vold = v; v = vnew;
    end
    data[:,i+1] = v; tdata = [tdata; t];
    if norm(v,Inf)>2.5
        data = data[:,1:i+1];
        break
    end
end

# Plot results:
mesh(x,tdata,data',ccount=0);
xlim(0,2*pi); ylim(0,tmax); zlim(-3,3);
xlabel("x"); ylabel("t"); zlabel("u"); gca()[:view_init](70,10-90);
