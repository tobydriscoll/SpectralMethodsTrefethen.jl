# p4.jl - periodic spectral differentiation
using PyPlot;  clf();
# Set up grid and differentiation matrix:
N = 24; h = 2*pi/N; x = h*(1:N);
column = [0; @. .5*(-1)^(1:N-1)*cot((1:N-1)*h/2)];
D = toeplitz(column,column[[1;N:-1:2]]);

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
