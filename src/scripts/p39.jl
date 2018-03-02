# p39.m - eigenmodes of biharmonic on a square with clamped BCs
#         (compare p38.m)

using SMiJ, PyPlot, Interpolations
# Construct spectral approximation to biharmonic operator:
N = 17; (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
S = diagm([0; 1 ./(1-x[2:N].^2); 0]);
D4 = (diagm(1-x.^2)*D^4 - 8*diagm(x)*D^3 - 12*D^2)*S;
D4 = D4[2:N,2:N]; I = eye(N-1);
L = kron(I,D4) + kron(D4,I) + 2*kron(D2,I)*kron(I,D2);

# Find and plot 25 eigenmodes:
(Lam,V) = eig(-L); Lam = -real(Lam);
ii = sortperm(Lam)[1:25]; Lam = Lam[ii]; V = real(V[:,ii]);
Lam = sqrt.(Lam/Lam[1]);
y = x; xxx = yyy = -1:.01:1;
sq = [1+1im,-1+1im,-1-1im,1-1im,1+1im]; clf();
for i = 1:25
    uu = zeros(N+1,N+1); uu[2:N,2:N] = reshape(V[:,i],N-1,N-1);
    subplot(5,5,i); plot(real(sq),imag(sq));
    s = interpolate((x[end:-1:1],y[end:-1:1]),reduce(flipdim,uu,1:2),Gridded(Linear()));
    contour(xxx,yyy,s[xxx,yyy],levels=[0],color="k"); axis("square");
    axis(1.25*[-1,1,-1,1]); axis("off");
    text(-.3,1.15,"$(signif(Lam[i],5))",fontsize=7);
end
