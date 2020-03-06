# p5.jl - repetition of p4.jl via FFT
#        For complex v, delete "real" commands.

# Differentiation of a hat function:
N = 24
h = 2π/N
x = h*(1:N)
v = @. max(0,1-abs(x-pi)/2)
v̂ = fft(v)
ŵ = 1im*[0:N/2-1;0;-N/2+1:-1] .* v̂
w = real(ifft(ŵ))

plt = plot(layout=(2,2),grid=true,xlim=(0,2π))
plot!(x,v,m=4,subplot=1,ylim=(-0.5,1.5),title="function")
plot!(x,D*v,m=4,subplot=2,ylim=(-1,1),title="spectral derivative")

# Differentiation of exp(sin(x)):
v = @. exp(sin(x))
vprime = @. cos(x)*v
v̂ = fft(v)
ŵ = 1im*[0:N/2-1;0;-N/2+1:-1] .* v̂
w = real(ifft(ŵ))

plot!(x,v,m=4,subplot=3,ylim=(0,3))
plot!(x,D*v,m=4,subplot=4,ylim=(-2,2))
str = @sprintf("max error = %0.4e",norm(D*v-vprime,Inf))
annotate!(2.2,1.4,text(str,8,:left),subplot=4)
