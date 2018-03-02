# p21.jl - eigenvalues of Mathieu operator -u_xx + 2qcos(2x)u
#         (compare p8.jl and p. 724 of Abramowitz & Stegun)

using PyPlot
N = 42; h = 2*pi/N; x = h*(1:N);
D2 = toeplitz([-pi^2/(3*h^2)-1/6; @. -.5*(-1)^(1:N-1)/sin(h*(1:N-1)/2)^2]);
qq = 0:.2:15; data = [];
for q = qq;
    e = sort(eigvals(-D2 + 2*q*diagm(cos.(2*x))));
    data = [data; e[1:11]'];
end
clf(); subplot(1,2,1);
plot(qq,data[:,1:2:end],"b-");
plot(qq,data[:,2:2:end],"b--");
xlabel("q"); ylabel("λ");
axis([0,15,-24,32]); yticks(-24:4:32);
