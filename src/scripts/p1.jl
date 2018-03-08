# p1.jl - convergence of fourth-order finite differences

# For various N, set up grid in [-pi,pi] and function u(x):
Nvec = 2.^(3:12);
clf(); axes([.1,.4,.8,.5]);
for N = Nvec
    h = 2*pi/N; x = -pi + (1:N)*h;
    u = @. exp(sin(x)^2);
    uprime = @. 2*sin(x)*cos(x)*u;

    # Construct sparse fourth-order differentiation matrix:
    e = ones(N);
    D = sparse(1:N,[2:N;1],2*e/3) - sparse(1:N,[3:N;1:2],e/12);
    D = (D-D')/h;

    # Plot max(abs(D*u-uprime)):
    error = norm(D*u-uprime,Inf);
    loglog(N,error,"k.",markersize=6);
end
grid(true); xlabel("N"); ylabel("error");
title("Convergence of fourth-order finite differences");
loglog(Nvec,1.0./Nvec.^4,"--");
text(105,5e-8,L"N^{-4}",fontsize=18);
