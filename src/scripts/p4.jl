# p4.jl - periodic spectral differentiation
using PyPlot;  clf();
# Set up grid and differentiation matrix:
N = 24; h = 2*pi/N; x = h*(1:N);
D = [ 0.5*(-1)^(i-j)*cot((i-j)*h/2) for i=1:N, j=1:N ];
D[1:N+1:end] = 0;

# Differentiation of a hat function:
v = @. max(0,1-abs(x-pi)/2);
subplot(2,2,1);  plot(x,v,".-",markersize=10);
axis([0,2pi,-.5,1.5]); grid(true); title("function");
subplot(2,2,2), plot(x,D*v,".-",markersize=10);
axis([0,2pi,-1,1]); grid(true); title("spectral derivative");

# Differentiation of exp(sin(x)):
v = @. exp(sin(x)); vprime = @. cos(x)*v;
subplot(2,2,3), plot(x,v,".-",markersize=10);
axis([0,2pi,0,3]), grid(true);
subplot(2,2,4), plot(x,D*v,".-",markersize=10);
axis([0,2pi,-2,2]), grid(true);
error = signif(norm(D*v-vprime,Inf),4);
text(2.2,1.4,"max error = $error",fontsize=8);
