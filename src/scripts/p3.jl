# p3.jl - band-limited interpolation

using PyPlot;  clf();
h = 1; xmax = 10;
x = -xmax:h:xmax;                     # computational grid
xx = -xmax-h/20:h/10:xmax+h/20;       # plotting grid
v = zeros(length(x),3);
v[:,1] = @. float(x==0);
v[:,2] = @. float(abs(x) ≤ 3);
v[:,3] = @. max(0,1-abs(x)/3);
for plt = 1:3
    subplot(3,1,plt);
    plot(x,v[:,plt],".",markersize=12);
    p = sum( v[i,plt]*sin.(pi*(xx-x[i])/h)./(pi*(xx-x[i])/h) for i=1:length(x) );
    plot(xx,p,"-")
    xlim(-xmax,xmax);  ylim(-0.5,1.5);
    xticks(1:0);  yticks(0:1);
end
