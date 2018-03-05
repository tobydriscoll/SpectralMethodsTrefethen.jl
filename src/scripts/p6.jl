# p6.jl - variable coefficient wave equation

# Grid, variable coefficient, and initial data:
N = 128; h = 2*pi/N; x = h*(1:N); t = 0; dt = h/4;
c = @. .2 + sin(x-1)^2;
v = @. exp(-100*(x-1).^2); vold = @. exp(-100*(x-.2*dt-1).^2);

# Time-stepping by leap frog formula:
tmax = 8; tplot = .15;
clf();
plotgap = round(tplot/dt); dt = tplot/plotgap;
nplots = Integer(round(tmax/tplot));
data = [v zeros(N,nplots)]; tdata = t;
for i = 1:nplots
    for n = 1:plotgap
      t = t+dt;
      v_hat = fft(v);
      w_hat = 1im*[0:N/2-1;0;-N/2+1:-1] .* v_hat;
      w = real(ifft(w_hat));
      vnew = vold - 2*dt*c.*w; vold = v; v = vnew;
    end
    data[:,i+1] = v; tdata = [tdata; t];
end
mesh(x,tdata,data',ccount=0);
xlim(0,2*pi); ylim(0,tmax); zlim(0,5);
xlabel("x"); ylabel("t"); zlabel("u"); gca()[:view_init](70,10-90);
