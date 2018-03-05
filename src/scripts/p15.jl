# p15.jl - solve eigenvalue BVP u_xx = lambda*u, u(-1)=u(1)=0

N = 36; (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
(lam,V) = eig(D2);
ii = sortperm(-lam);          # sort eigenvalues and -vectors
lam = lam[ii]; V = V[:,ii];
clf();
for j = 5:5:30                  # plot 6 eigenvectors
    u = [0;V[:,j];0]; subplot(6,1,j/5)
    plot(x,u,".",markersize=8); grid(true);
    xx = -1:.01:1; uu = polyval(polyfit(x,u),xx);
    plot(xx,uu); axis("off");
    text(-.4,.1,"eig $j = $(lam[j]*4/pi^2) π^2/4");
    text(.7,.1,"$(signif(4*N/(pi*j),2))  ppw");
end
