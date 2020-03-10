# p26.jl - eigenvalues of 2nd-order Chebyshev diff. matrix

N = 60
D,x = cheb(N)
D2 = D^2
D2 = D2[2:N,2:N]
λ,V = eigen(-D2)

plt = plot(layout = grid(3,1,heights=[0.5,0.25,0.25]))
# Plot eigenvalues:
str = "N = $N       max |\\lambda| = $(round(maximum(λ)/N^4,sigdigits=5)) N^4"
scatter!(λ,subplot=1,xaxis=:log,yaxis=(:log,"eigenvalue"),title=str)
plot!(2*N/π*[1,1],[1,1e6],l=(:dash,:red),subplot=1)
annotate!(1.9*N/π,24,text(L"2\pi / N",:right,9),subplot=1)

# Plot eigenmodes N/4 (physical) and N (nonphysical):
xx = -1:.01:1
vN4 = [0; V[:,Int(N/4-1)]; 0]
vv = polyval(polyfit(x,vN4),xx)
plot!(xx,vv,subplot=2,yaxis=[-0.2,0,0.2])
scatter!(x,vN4,subplot=2,title="eigenmode N/4")

vN = V[:,N-1];
plot!(x[2:N],abs.(vN),m=4,subplot=3,yaxis=(:log,10.0.^[-5,-3,-1]),title="absolute value of eigenmode N  (log scale)")
