# p22.jl - 5th eigenvector of Airy equation u_xx = lambda*x*u

clf();
for N = 12:12:48
    (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
    (lam,V) = eig(D2,diagm(x[2:N]));      # generalized ev problem
    ii = find(lam.>0);
    V = V[:,ii]; lam = lam[ii];
    ii = sortperm(lam)[5]; lambda = lam[ii];
    v = [0;V[:,ii];0]; v = v/v[Int(N/2+1)]*airyai(0);
    xx = -1:.01:1; vv = polyval(polyfit(x,v),xx);
    subplot(2,2,N/12); plot(xx,vv); grid(true)
    title("N = $N     eig = $(signif(lambda,13))");
end
