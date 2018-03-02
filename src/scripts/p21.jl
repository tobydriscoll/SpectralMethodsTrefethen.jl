# p21.jl - eigenvalues of Mathieu operator -u_xx + 2qcos(2x)u
#         (compare p8.jl and p. 724 of Abramowitz & Stegun)

using PyPlot
N = 42; h = 2*pi/N; x = h*(1:N);
D2 = [ -.5*(-1)^(i-j)/sin(h*(i-j)/2)^2 for i = 0:N-1, j = 0:N-1 ];
D2[1:N+1:end] = -pi^2/(3*h^2)-1/6;
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
