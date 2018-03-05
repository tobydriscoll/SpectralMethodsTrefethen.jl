# p31.jl - gamma function via complex integral, trapezoid rule

N = 70; theta = -pi + (2*pi/N)*(.5:N-.5);
c = -11;                     # center of circle of integration
r = 16;                      # radius of circle of integration
x = -3.5:.1:4; y = -2.5:.1:2.5;
zz = x' .+ 1im*y; gaminv = 0*zz;
for i = 1:N
    t = c + r*exp(1im*theta[i]);
    gaminv += exp(t)*t.^(-zz)*(t-c);
end
gaminv = gaminv/N; gam = 1./gaminv;
clf(); surf(x,y,abs.(gam));
xlim(-3.5,4); ylim(-2.5,2.5); zlim(0,6);
xlabel("Re(z)"); ylabel("Im(z)");
text3D(4,-1.4,5.5,"\$|\\Gamma(z)|\$",fontsize=20);
