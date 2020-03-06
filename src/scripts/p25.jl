# p25.jl - stability regions for ODE formulas

plt = plot(layout=(2,2),aspect_ratio=1,framestyle=:origin)
ReIm = z->(real(z),imag(z))
z = [ exp(1im*π*x) for x in (0:200)/100 ]

# Adams-Bashforth, orders 1-3:
r = z.-1
for f in [z->1,z->(3-1/z)/2,z->(23-16/z+5/z^2)/12]
    plot!(ReIm(@. r/f(z))...,subplot=1)
end
plot!(subplot=1,title="Adams-Bashforth")

# Adams-Moulton, orders 3-6:
sfun = [z->(5*z+8-1/z)/12,
        z->(9*z+19-5/z+1/z^2)/24,
        z->(251*z+646-264/z+106/z^2-19/z^3)/720 ]
for f in sfun
    plot!(ReIm(@. r/f(z))...,subplot=2)
end
d = @. 1-1/z
p = Poly([1,-1/2,-1/12,-1/24,-19/720,-3/160])
plot!(ReIm(@. d/p(d))...,subplot=2,title="Adams-Moulton")

# Backward differentiation, orders 1-6:
r = zeros(size(d))
for i in 1:6
    global r += (d.^i)/i
    plot!(ReIm(r)...,subplot=3)
end
plot!(subplot=3,title="backward differentiation")

# Runge-Kutta, orders 1-4:
w = complex(zeros(4))
W = w
for i in 2:length(z)
    w[1] -= (1+w[1]-z[i])
    w[2] -= (1+w[2]+.5*w[2]^2-z[i]^2)/(1+w[2])
    w[3] -= (1+w[3]+.5*w[3]^2+w[3]^3/6-z[i]^3)/(1+w[3]+w[3]^2/2)
    w[4] -= (1+w[4]+.5*w[4]^2+w[4]^3/6+w[4].^4/24-z[i]^4)/(1+w[4]+w[4]^2/2+w[4]^3/6)
    global W = [W w]
end
plot!(ReIm(transpose(W))...,subplot=4,title="Runge-Kutta")
