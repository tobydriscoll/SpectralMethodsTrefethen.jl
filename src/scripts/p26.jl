# p26.m - eigenvalues of 2nd-order Chebyshev diff. matrix

N = 60; (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
(lam,V) = eig(D2);
ii = sortperm(-lam); e = lam[ii]; V = V[:,ii];

# Plot eigenvalues:
clf(); axes([.1, .62, .8, .3]);
loglog(-e,".",markersize=4); ylabel("eigenvalue");
title("N = $N       max |λ| = $(signif(maximum(-e)/N^4,5)) \$N^4\$");
semilogy(2*N/pi*[1,1],[1,1e6],"--r");
text(2.1*N/pi,24,"2π / N",fontsize=12);

# Plot eigenmodes N/4 (physical) and N (nonphysical):
vN4 = [0; V[:,Int(N/4-1)]; 0];
xx = -1:.01:1; vv = polyval(polyfit(x,vN4),xx);
axes([.1, .36, .8, .15]); plot(xx,vv);
plot(x,vN4,"k.",markersize=4); title("eigenmode N/4");
vN = V[:,N-1];
axes([.1, .1, .8, .15]);
semilogy(x[2:N],abs.(vN)); axis([-1,1,5e-6,1]);
plot(x[2:N],abs.(vN),"k.",markersize=4);
title("absolute value of eigenmode N    (log scale)");
