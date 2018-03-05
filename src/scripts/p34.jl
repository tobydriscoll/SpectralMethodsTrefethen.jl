# p34.jl - Allen-Cahn eq. u_t = eps*u_xx+u-u^3, u(-1)=-1, u(1)=1
#         (compare p6.jl and p32.jl)

# Differentiation matrix and initial data:
N = 20; (D,x) = cheb(N); D2 = D^2;     # use full-size matrix
D2[[1,N+1],:] = 0;                     # for convenience
eps = 0.01; dt = min(.01,50/(N^4*eps));
t = 0.0; v = @. .53*x + .47*sin(-1.5*pi*x);

# Solve PDE by Euler formula and plot results:
tmax = 100; tplot = 2; nplots = round(Int,tmax/tplot);
plotgap = round(Int,tplot/dt); dt = tplot/plotgap;
xx = -1:.025:1; vv = polyval(polyfit(x,v),xx);
plotdata = [vv zeros(length(xx),nplots)]; tdata = t;
for i = 1:nplots
    for n = 1:plotgap
        t = t+dt; v = v + dt*(eps*D2*(v-x) + v - v.^3);    # Euler
    end
    vv = polyval(polyfit(x,v),xx);
    plotdata[:,i+1] = vv; tdata = [tdata; t];
end
clf();
surf(xx,tdata,plotdata'); grid(true);
xlim(-1,1); ylim(0,tmax); zlim(-1,1);
gca()[:view_init](55,-150);
xlabel("x"); ylabel("t"); zlabel("u");
