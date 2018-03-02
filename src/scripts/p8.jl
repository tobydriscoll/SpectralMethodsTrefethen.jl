# p8.m - eigenvalues of harmonic oscillator -u"+x^2 u on R

L = 8;                             # domain is [-L L], periodic
for N = 6:6:36
    h = 2*pi/N; x = h*(1:N); x = L*(x-pi)/pi;
    column = [-pi^2/(3*h^2)-1/6; @. -.5*(-1)^(1:N-1)/sin(h*(1:N-1)/2)^2 ];
    D2 = (pi/L)^2*toeplitz(column);  # 2nd-order differentiation
    eigenvalues = sort(eigvals(-D2 + diagm(x.^2)));
    @show N
    [ println(eigenvalues[i]) for i=1:4 ];
    println("");
end
