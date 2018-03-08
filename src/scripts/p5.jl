# p5.jl - repetition of p4.jl via FFT
#        For complex v, delete "real" commands.

# Differentiation of a hat function:
N = 24; h = 2*pi/N; x = h*(1:N);
v = @. max(0,1-abs(x-pi)/2); v_hat = fft(v);
w_hat = 1im*[0:N/2-1;0;-N/2+1:-1] .* v_hat;
w = real(ifft(w_hat)); clf();
subplot(2,2,1);  plot(x,v,".-",markersize=6);
axis([0,2pi,-.5,1.5]); grid(true); title("function");
subplot(2,2,2), plot(x,D*v,".-",markersize=6);
axis([0,2pi,-1,1]); grid(true); title("spectral derivative");

# Differentiation of exp(sin(x)):
v = @. exp(sin(x)); vprime = @. cos(x)*v;
v_hat = fft(v);
w_hat = 1im*[0:N/2-1;0;-N/2+1:-1] .* v_hat;
w = real(ifft(w_hat));
subplot(2,2,3), plot(x,v,".-",markersize=6);
axis([0,2pi,0,3]), grid(true);
subplot(2,2,4), plot(x,D*v,".-",markersize=6);
axis([0,2pi,-2,2]), grid(true);
error = signif(norm(D*v-vprime,Inf),4);
text(2.2,1.4,"max error = $error",fontsize=8);
