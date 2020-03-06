# p3.jl - band-limited interpolation

h = 1
xmax = 10
x = -xmax:h:xmax                     # computational grid
xx = -xmax-h/20:h/10:xmax+h/20       # plotting grid
fun = [ x->x==0, x->abs(x) ≤ 3, x->max(0,1-abs(x)/3) ]
sinc = t -> sin(π*t/h)/(π*t/h)

plt = plot(layout=(3,1),xaxis=((-xmax,xmax),[]),yaxis=((-0.5,1.5),[0,0,1]))
for i in 1:3
    v = fun[i].(x)
    scatter!(x,v,subplot=i)
    p = sum( v[j]*sinc.(xx.-x[j]) for j in eachindex(x) )
    plot!(xx,p,subplot=i)
end
