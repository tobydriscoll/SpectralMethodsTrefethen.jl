# p4.jl - periodic spectral differentiation

# Set up grid and differentiation matrix:
N = 24
h = 2π/N
x = h*(1:N)
column = [0; [.5*(-1)^j*cot(j*h/2) for j in 1:N-1]  ]
D = toeplitz(column,column[[1;N:-1:2]])

plt = plot(layout=(2,2))
# Differentiation of a hat function:
v = @. max(0,1-abs(x-pi)/2)
plot!(x,v,subplot=1,m=true,
	xaxis=(0,2π),yaxis=(-0.5,1.5),title="function")
plot!(x,D*v,subplot=2,m=true,
	xaxis=(0,2π),yaxis=(-1,1),title="spectral derivative")

# Differentiation of exp(sin(x)):
v = @. exp(sin(x))
vprime = @. cos(x)*v
plot!(x,v,subplot=3,m=true,xaxis=(0,2π),yaxis=(0,3))
plot!(x,D*v,subplot=4,m=true,xaxis=(0,2π),yaxis=(-2,2))
str = @sprintf("max error = %0.4e",norm(D*v-vprime,Inf))
annotate!(2.2,1.4,text(str,8,:left),subplot=4)
