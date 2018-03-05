# p25.jl - stability regions for ODE formulas

# Adams-Bashforth:
clf(); subplot(221);
zplot(z) = plot(real(z),imag(z));
plot([-8,8],[0,0],"k-"); plot([0,0],[-8,8],"k-");
z = exp.(1im*pi*(0:200)/100); r = z-1;
s = 1; zplot(r./s);                                  # order 1
s = (3-1./z)/2; zplot(r./s);                         # order 2
s = (23-16./z+5./z.^2)/12; zplot(r./s);             # order 3
axis("square"); axis([-2.5,.5,-1.5,1.5]); grid(true)
title("Adams—Bashforth");

# Adams-Moulton:
subplot(222);
plot([-8,8],[0,0],"k-"); plot([0,0],[-8,8],"k-");
s = (5*z+8-1./z)/12; zplot(r./s);                    # order 3
s = (9*z+19-5./z+1./z.^2)/24; zplot(r./s);           # order 4
s = (251*z+646-264./z+106./z.^2-19./z.^3)/720; zplot(r./s) ;   # 5
d = 1-1./z;
s = 1-d/2-d.^2/12-d.^3/24-19*d.^4/720-3*d.^5/160; zplot(d./s); # 6
axis("square"); axis([-7,1,-4,4]); grid(true);
title("Adams—Moulton");

# Backward differentiation:
subplot(223);
plot([-40,40],[0,0],"k-"); plot([0,0],[-40,40],"k-");
r = 0;
for i = 1:6  # orders 1-6
    r += (d.^i)/i;
    zplot(r);
end
axis("square"); axis([-15,35,-25,25]); grid(true);
title("backward differentiation")

# Runge-Kutta:
subplot(224);
plot([-8,8],[0,0],"k-"); plot([0,0],[-8,8],"k-");
w = zeros(4)+0im; W = w.';
for i = 2:length(z)
    # orders 1-4
    w[1] -= (1+w[1]-z[i]);
    w[2] -= (1+w[2]+.5*w[2]^2-z[i]^2)/(1+w[2]);
    w[3] -= (1+w[3]+.5*w[3]^2+w[3]^3/6-z[i]^3)/(1+w[3]+w[3]^2/2);
    w[4] -= (1+w[4]+.5*w[4]^2+w[4]^3/6+w[4].^4/24-z[i]^4)/(1+w[4]+w[4]^2/2+w[4]^3/6);
    W = [W; w.'];
end
zplot(W); axis("square"); axis([-5,2,-3.5,3.5]); grid(true);
title("Runge—Kutta");
