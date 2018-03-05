# p40.jl - eigenvalues of Orr-Sommerfeld operator (compare p38.jl)

R = 5772; clf();
for N = 40:20:100
    # 2nd- and 4th-order differentiation matrices:
    (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
    S = diagm([0; 1 ./(1-x[2:N].^2); 0]);
    D4 = (diagm(1-x.^2)*D^4 - 8*diagm(x)*D^3 - 12*D^2)*S;
    D4 = D4[2:N,2:N];

    # Orr-Sommerfeld operators A,B and generalized eigenvalues:
    I = eye(N-1);
    A = (D4-2*D2+I)/R - 2im*I - 1im*diagm(1-x[2:N].^2)*(D2-I);
    B = D2-I;
    ee = eigvals(A,B);
    i = Int(N/20-1); subplot(2,2,i);
    plot(real(ee),imag(ee),".",markersize=8);
    grid(true); axis("square"); axis([-.8,.2,-1,0]);
    title("N = $N    \$\\lambda_{max}\$ = $(signif(maximum(real(ee)),7))");
end
