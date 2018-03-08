# p3.jl - band-limited interpolation

h = 1; xmax = 10; clf();
x = -xmax:h:xmax;                     # computational grid
xx = -xmax-h/20:h/10:xmax+h/20;       # plotting grid
v = zeros(length(x),3);
v[:,1] = @. float(x==0);
v[:,2] = @. float(abs(x) ≤ 3);
v[:,3] = @. max(0,1-abs(x)/3);
for plt = 1:3
    subplot(4,1,plt);
    plot(x,v[:,plt],".",markersize=6);
    p = 0;
    for i = 1:length(x)
        p = @. p + v[i,plt]*sin(pi*(xx-x[i])/h)/(pi*(xx-x[i])/h);
    end
    plot(xx,p,"-")
    axis([-xmax,xmax,-0.5,1.5]);
    xticks(1:0); yticks(0:1);
end
