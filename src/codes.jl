function p1()
    # p1.jl - convergence of fourth-order finite differences
    
    # For various N, set up grid in [-pi,pi] and function u(x):
    Nvec = @. 2^(3:12);
    clf(); PyPlot.axes([.1,.4,.8,.5]);
    for N = Nvec
        h = 2*pi/N
        x = @. -pi + (1:N)*h;
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
end

function p2()
    # p2.jl - convergence of periodic spectral method (compare p1.jl)
    
    # For various N (even), set up grid as before:
    clf(); PyPlot.axes([.1,.4,.8,.5]);
    for N = 2:2:100
        h = 2*pi/N;  x = [ -pi + i*h for i = 1:N ];
        u = @. exp(sin(x)); uprime = @. cos(x)*u;
    
        # Construct spectral differentiation matrix:
        column = [ 0; @. .5*(-1)^(1:N-1)*cot((1:N-1)*h/2) ];
        D = toeplitz(column,column[[1;N:-1:2]]);
    
        # Plot max(abs(D*u-uprime)):
        error = norm(D*u-uprime,Inf);
        loglog(N,error,"k.",markersize=6);
    end
    grid(true); xlabel("N"); ylabel("error");
    title("Convergence of spectral differentiation")
end

function p3()
    # p3.jl - band-limited interpolation
    
    h = 1; xmax = 10; clf();
    x = -xmax:h:xmax;                     # computational grid
    xx = -xmax-h/20:h/10:xmax+h/20;       # plotting grid
    v = zeros(length(x),3);
    v[:,1] = @. float(x==0);
    v[:,2] = @. float(abs(x) ≤ 3);
    v[:,3] = @. max(0,1-abs(x)/3);
    for plt = 1:3
        subplot(4,1,plt);
        plot(x,v[:,plt],".",markersize=6);
        p = 0;
        for i = 1:length(x)
            p = @. p + v[i,plt]*sin(pi*(xx-x[i])/h)/(pi*(xx-x[i])/h);
        end
        plot(xx,p,"-")
        axis([-xmax,xmax,-0.5,1.5]);
        xticks(1:0); yticks(0:1);
    end
end

function p4()
    # p4.jl - periodic spectral differentiation
    
    # Set up grid and differentiation matrix:
    N = 24; h = 2*pi/N; x = h*(1:N);
    column = [0; @. .5*(-1)^(1:N-1)*cot((1:N-1)*h/2)];
    D = toeplitz(column,column[[1;N:-1:2]]);
    
    # Differentiation of a hat function:
    v = @. max(0,1-abs(x-pi)/2); clf();
    subplot(2,2,1);  plot(x,v,".-",markersize=6);
    axis([0,2pi,-.5,1.5]); grid(true); title("function");
    subplot(2,2,2), plot(x,D*v,".-",markersize=6);
    axis([0,2pi,-1,1]); grid(true); title("spectral derivative");
    
    # Differentiation of exp(sin(x)):
    v = @. exp(sin(x)); vprime = @. cos(x)*v;
    subplot(2,2,3), plot(x,v,".-",markersize=6);
    axis([0,2pi,0,3]), grid(true);
    subplot(2,2,4), plot(x,D*v,".-",markersize=6);
    axis([0,2pi,-2,2]), grid(true);
    error = round(norm(D*v-vprime,Inf),sigdigits=5);
    text(2.2,1.4,"max error = $error",fontsize=8);
end

function p5()
    # p5.jl - repetition of p4.jl via FFT
    #        For complex v, delete "real" commands.
    
    # Differentiation of a hat function:
    N = 24; h = 2*pi/N; x = h*(1:N);
    v = @. max(0,1-abs(x-pi)/2); v_hat = fft(v);
    w_hat = 1im*[0:N/2-1;0;-N/2+1:-1] .* v_hat;
    w = real(ifft(w_hat)); clf();
    subplot(2,2,1);  plot(x,v,".-",markersize=6);
    axis([0,2pi,-.5,1.5]); grid(true); title("function");
    subplot(2,2,2), plot(x,D*v,".-",markersize=6);
    axis([0,2pi,-1,1]); grid(true); title("spectral derivative");
    
    # Differentiation of exp(sin(x)):
    v = @. exp(sin(x)); vprime = @. cos(x)*v;
    v_hat = fft(v);
    w_hat = 1im*[0:N/2-1;0;-N/2+1:-1] .* v_hat;
    w = real(ifft(w_hat));
    subplot(2,2,3), plot(x,v,".-",markersize=6);
    axis([0,2pi,0,3]), grid(true);
    subplot(2,2,4), plot(x,D*v,".-",markersize=6);
    axis([0,2pi,-2,2]), grid(true);
    error = round(norm(D*v-vprime,Inf),sigdigits=4);
    text(2.2,1.4,"max error = $error",fontsize=8);
end

function p6()
    # p6.jl - variable coefficient wave equation
    
    # Grid, variable coefficient, and initial data:
    N = 128; h = 2*pi/N; x = h*(1:N); t = 0; dt = h/4;
    c = @. .2 + sin(x-1)^2;
    v = @. exp(-100*(x-1).^2); vold = @. exp(-100*(x-.2*dt-1).^2);
    
    # Time-stepping by leap frog formula:
    tmax = 8; tplot = .15; clf();
    plotgap = round(tplot/dt); dt = tplot/plotgap;
    nplots = round(Int,tmax/tplot);
    data = [v zeros(N,nplots)]; tdata = [t];
    for i = 1:nplots
      global data,tdata
        for n = 1:plotgap
          global t = t+dt;
          global v
          global vold
          v_hat = fft(v);
          w_hat = 1im*[0:N/2-1;0;-N/2+1:-1] .* v_hat;
          w = real(ifft(w_hat));
          vnew = vold - 2*dt*c.*w; vold = v; v = vnew;
        end
        data[:,i+1] = v;
        tdata = [tdata; t];
    end
    mesh(x,tdata,data',ccount=0); view(10,70);
    xlim(0,2*pi); ylim(0,tmax); zlim(0,5);
    xlabel("x"); ylabel("t"); zlabel("u");
end

function p6u()
    # p6u.jl - variable coefficient wave equation - UNSTABLE VARIANT
    
    # Grid, variable coefficient, and initial data:
    N = 128; h = 2*pi/N; x = h*(1:N);
    c = @. .2 + sin(x-1)^2;
    t = 0; dt = 1.9/N;
    v = @. exp(-100*(x-1)^2);
    vold = @. exp(-100*(x-.2*dt-1)^2);
    
    # Time-stepping by leap frog formula:
    tmax = 8; tplot = .15; clf();
    plotgap = round(Int,tplot/dt);
    nplots = round(Int,tmax/tplot);
    data = [v zeros(N,nplots)]; tdata = t;
    for i = 1:nplots
        for n = 1:plotgap
            t = t+dt;
            v_hat = fft(v);
            w_hat = 1im*[0:N/2-1; 0; -N/2+1:-1] .* v_hat;
            w = real.(ifft(w_hat));
            vnew = vold - 2*dt*c.*w;        # leap frog formula
            vold = v; v = vnew;
        end
        data[:,i+1] = v; tdata = [tdata; t];
        if norm(v,Inf)>2.5
            data = data[:,1:i+1];
            break
        end
    end
    
    # Plot results:
    mesh(x,tdata,data',ccount=0);
    xlim(0,2*pi); ylim(0,tmax); zlim(-3,3);
    xlabel("x"); ylabel("t"); zlabel("u"); gca()[:view_init](70,10-90);
end

function p7()
    # p7.m - accuracy of periodic spectral differentiation
    
    # Compute derivatives for various values of N:
    Nmax = 50; allN = 6:2:Nmax; E = zeros(4,length(allN));
    for N = 6:2:Nmax
        h = 2*pi/N; x = h*(1:N);
        column = [0; @. .5*(-1)^(1:N-1)*cot((1:N-1)*h/2)];
        D = toeplitz(column,column[[1;N:-1:2]]);
        v = @. abs(sin(x))^3;                     # 3rd deriv in BV
        vprime = @. 3*sin(x)*cos(x)*abs(sin(x));
        j = round(Int,N/2-2);
        E[1,j] = norm(D*v-vprime,Inf);
        v = @. exp(-sin(x/2)^(-2));               # C-infinity
        vprime = @. .5*v*sin(x)/sin(x/2)^4;
        E[2,j] = norm(D*v-vprime,Inf);
        v = @. 1/(1+sin(x/2)^2);                 # analytic in a strip
        vprime = @. -sin(x/2)*cos(x/2)*v^2;
        E[3,j] = norm(D*v-vprime,Inf);
        v = sin.(10*x); vprime = 10*cos.(10*x);   # band-limited
        E[4,j] = norm(D*v-vprime,Inf);
    end
    
    # Plot results:
    titles = [L"|\sin(x)|^3",L"\exp(-\sin^{-2}(x/2))",L"1/(1+\sin^2(x/2))",L"\sin(10x)"]; clf();
    for iplot = 1:4
        subplot(2,2,iplot);
        semilogy(allN,E[iplot,:],".-",markersize=6);
        axis([0,Nmax,1e-16,1e3]); grid(true);
        xticks(0:10:Nmax); yticks(10.0.^(-15:5:0));
        title(titles[iplot]);
        iplot > 2 ? xlabel("N") : nothing;
        iplot%2 > 0 ? ylabel("error") : nothing;
    end
end

function p8()
    # p8.m - eigenvalues of harmonic oscillator -u"+x^2 u on R
    
    L = 8;                             # domain is [-L L], periodic
    for N = 6:6:36
        h = 2*pi/N; x = h*(1:N); x = @. L*(x-pi)/pi;
        column = [-pi^2/(3*h^2)-1/6; @. -.5*(-1)^(1:N-1)/sin(h*(1:N-1)/2)^2 ];
        D2 = (pi/L)^2*toeplitz(column);  # 2nd-order differentiation
        eigenvalues = sort(eigvals(-D2 + diagm(x.^2)));
        @show N
        [ println(eigenvalues[i]) for i=1:4 ];
        println("");
    end
end

function p9()
    # p9.jl - polynomial interpolation in equispaced and Chebyshev pts
    
    N = 16;
    xx = -1.01:.005:1.01; clf();
    for i = 1:2
        i==1 && ( (s,x) = ("equispaced points", -1 .+ 2*(0:N)/N) );
        i==2 && ( (s,x) = ("Chebyshev points", cos.(pi*(0:N)/N)) );
        subplot(2,2,i)
        u = @. 1/(1+16*x^2);
        uu = @. 1/(1+16*xx^2);
        p = fit(x,u);              # interpolation
        pp = p.(xx);                    # evaluation of interpolant
        plot(x,u,".",markersize=6)
        plot(xx,pp)
        axis([-1.1,1.1,-1,1.5]); title(s);
        error = round( norm(uu-pp,Inf), sigdigits=5);
        text(-.5,-.5,"max error = $error",fontsize=8);
    end
end

function p10()
    # p10.jl - polynomials and corresponding equipotential curves
    
    N = 16; clf();
    xx = -1.01:.005:1.01;
    for i = 1:2
        i==1 && ( (s,x) = ("equispaced points", -1 .+ 2*(0:N)/N) );
        i==2 && ( (s,x) = ("Chebyshev points", cos.(pi*(0:N)/N)) );
        p = fromroots(x);
    
        # Plot p(x) over [-1,1]:
        xx = -1:.005:1; pp = p.(xx);
        subplot(2,2,2*i-1);
        plot(x,0*x,"k.",markersize=6);
        plot(xx,pp);  grid(true);
        xticks(-1:.5:1);  title(s);
    
        # Plot equipotential curves:
        subplot(2,2,2*i)
        plot(real(x),imag(x),".",markersize=6);
        axis([-1.4,1.4,-1.12,1.12]);
        xx = -1.4:.02:1.4; yy = -1.12:.02:1.12;
        zz = xx' .+ 1im*yy;
        pp = p.(zz); levels = 10.0.^(-4:0);
        contour(xx,yy,abs.(pp),levels,colors="k"); title(s);
    end
end

function p11()
    # p11.jl - Chebyshev differentation of a smooth function
    
    xx = -1:.01:1; uu = @. exp(xx)*sin(5*xx); clf();
    for N = [10,20]
        D,x = cheb(N); 
        u = @. exp(x)*sin(5*x);
        PyPlot.axes([.15,.66-.4*(N==20),.31,.28]);
        plot(x,u,".",markersize=6); grid(true);
        plot(xx,uu);
        title("u(x),  N=$N")
        error = D*u - @. exp(x)*(sin(5*x)+5*cos(5*x));
        PyPlot.axes([.55,.66-.4*(N==20),.31,.28]);
        plot(x,error,".-",markersize=6); grid(true);
        title("error in u'(x),  N=$N");
    end
end

function p12()
    # p12.jl - accuracy of Chebyshev spectral differentiation
    #         (compare p7.jl)
    
    # Compute derivatives for various values of N:
    Nmax = 50; allN = 6:2:Nmax; E = zeros(4,Nmax);
    for N = 1:Nmax
        (D,x) = cheb(N);
        v = @. abs(x)^3; vprime = @. 3*x*abs(x);   # 3rd deriv in BV
        E[1,N] = norm(D*v-vprime,Inf);
        v = @. exp(-x^(-2)); vprime = @. 2*v/x^3;  # C-infinity
        E[2,N] = norm(D*v-vprime,Inf);
        v = @. 1/(1+x^2); vprime = @. -2*x*v^2;    # analytic in [-1,1]
        E[3,N] = norm(D*v-vprime,Inf);
        v = x.^10; vprime = 10*x.^9;               # polynomial
        E[4,N] = norm(D*v-vprime,Inf);
    end
    
    # Plot results:
    titles = [L"|x|^3",L"\exp(-x^2)",L"1/(1+x^2)",L"x^{10}"];  clf();
    for iplot = 1:4
        subplot(2,2,iplot);
        semilogy(1:Nmax,E[iplot,:],".-",markersize=6);
        axis([0,Nmax,1e-16,1e3]); grid(true);
        xticks(0:10:Nmax); yticks(10.0.^(-15:5:0));
        title(titles[iplot]);
        iplot > 2 ? xlabel("N") : nothing;
        iplot%2 > 0 ? ylabel("error") : nothing;
    end
end

function p13()
    # p13.jl - solve linear BVP u_xx = exp(4x), u(-1)=u(1)=0
    
    N = 16;
    (D,x) = cheb(N);
    D2 = D^2;
    D2 = D2[2:N,2:N];                   # boundary conditions
    f = @. exp(4*x[2:N]);
    u = D2\f;                           # Poisson eq. solved here
    u = [0;u;0];
    clf(); axes([.1,.4,.8,.5]);
    plot(x,u,".",markersize=6);
    xx = -1:.01:1;
    uu = fit(x,u).(xx);      # interpolate grid data
    plot(xx,uu);
    grid(true);
    exact = @. ( exp(4*xx) - sinh(4)*xx - cosh(4) )/16;
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=4))",fontsize=12);
end

function p14()
    # p14.jl - solve nonlinear BVP u_xx = exp(u), u(-1)=u(1)=0
    #         (compare p13.jl)
    
    N = 16;
    (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
    u = zeros(N-1);
    change = 1; it = 0;
    while change > 1e-15                   # fixed-point iteration
        unew = D2\exp.(u);
        global change = norm(unew-u,Inf);
        global u = unew; 
        global it = it+1;
    end
    u = [0;u;0];
    clf(); PyPlot.axes([.1,.4,.8,.5]);
    plot(x,u,".",markersize=6);
    xx = -1:.01:1;
    uu = fit(x,u).(xx);
    plot(xx,uu), grid(true);
    title("no. steps = $it      u(0) =$(u[Int(N/2+1)])")
end

function p15()
    # p15.jl - solve eigenvalue BVP u_xx = lambda*u, u(-1)=u(1)=0
    
    N = 36; (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
    lam,V = eigen(D2);
    ii = sortperm(-lam);          # sort eigenvalues and -vectors
    lam = lam[ii]; V = V[:,ii]; clf();
    for j = 5:5:30                  # plot 6 eigenvectors
        u = [0;V[:,j];0]; subplot(7,1,j÷5)
        plot(x,u,".",markersize=6); grid(true);
        xx = -1:.01:1; uu = fit(x,u).(xx);
        plot(xx,uu); axis("off");
        text(-.4,.1,"eig $j = $(lam[j]*4/pi^2) π^2/4");
        text(.7,.1,"$(round(4*N/(pi*j),sigdigits=2))  ppw");
    end
end

function p16()
    # p16.jl - Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary
    
    # Set up grids and tensor product Laplacian and solve for u:
    N = 24; (D,x) = cheb(N); y = x;
    xx = x[2:N]; yy = y[2:N];
    f = @. 10*sin(8*xx'*(yy-1));
    D2 = D^2; D2 = D2[2:N,2:N];
    L = kron(I(N-1),D2) + kron(D2,I(N-1));                       # Laplacian
    figure(1); clf(); spy(L);
    @elapsed u = L\f[:]           # solve problem and watch the clock
    
    # Reshape long 1D results onto 2D grid (flipping orientation):
    uu = zeros(N+1,N+1); uu[N:-1:2,N:-1:2] = reshape(u,N-1,N-1);
    value = uu[Int(3N/4+1),Int(3N/4+1)];
    
    # Interpolate to finer grid and plot:
    xxx = yyy = -1:.04:1;
    s = Spline2D(x[end:-1:1],y[end:-1:1],uu,kx=1,ky=1);
    uuu = s.(xxx,yyy');
    figure(2); clf(); surf(xxx,yyy,uuu,rstride=1,cstride=1);
    xlabel("x"); ylabel("y"); zlabel("u"); view(-37.5,30);
    text3D(.4,-.3,-.3,"\$u(2^{-1/2},2^{-1/2})\$ = $(round(value,sigdigits=11))",fontsize=9);
end

function p17()
    # p17.jl - Helmholtz eq. u_xx + u_yy + (k^2)u = f
    #         on [-1,1]x[-1,1]    (compare p16.jl)
    
    # Set up spectral grid and tensor product Helmholtz operator:
    N = 24; (D,x) = cheb(N); y = x;
    xx = x[2:N]; yy = y[2:N];
    f = @. exp(-10*((yy-1)^2+(xx'-.5)^2));
    D2 = D^2; D2 = D2[2:N,2:N];
    k = 9;
    L = kron(I(N-1),D2) + kron(D2,I(N-1)) + k^2*I((N-1)^2);
    
    # Solve for u, reshape to 2D grid, and plot:
    u = L\f[:];
    uu = zeros(N+1,N+1); uu[N:-1:2,N:-1:2] = reshape(u,N-1,N-1);
    xxx = yyy = -1:.0333:1;
    s = Spline2D(x[end:-1:1],y[end:-1:1],uu);
    uuu = evalgrid(s,xxx,yyy);
    figure(1); clf(); surf(xxx,yyy,uuu,rstride=1,cstride=1);
    xlabel("x"); ylabel("y"); zlabel("u"); view(-37.5,30);
    value = round(uu[Int(N/2+1),Int(N/2+1)],sigdigits=10);
    text3D(.2,1,.022,"u(0,0) = $value");
    figure(2); clf(); contour(xxx,yyy,uuu,10); axis("square");
end

function p18()
    # p18.jl - Chebyshev differentiation via FFT (compare p11.jl)
    
    xx = -1:.01:1; ff = @. exp(xx)*sin(5*xx); clf();
    for N = [10 20]
        x = cheb(N)[2]; f = @. exp(x)*sin(5*x);
        PyPlot.axes([.15, .66-.4*(N==20), .31, .28]);
        plot(x,f,"k.",markersize=6); grid(true);
        plot(xx,ff);
        title("f(x), N=$N");
        error = chebfft(f) - @. exp(x)*(sin(5*x)+5*cos(5*x));
        PyPlot.axes([.55, .66-.4*(N==20), .31, .28]);
        plot(x,error,".-",markersize=10); grid(true);
        title("error in f'(x),  N=$N");
    end
end

function p19()
    # p19.jl - 2nd-order wave eq. on Chebyshev grid (compare p6.jl)
    
    # Time-stepping by leap frog formula:
    N = 80; x = cheb(N)[2]; dt = 8/N^2;
    v = @. exp(-200*x^2); vold = @. exp(-200*(x-dt)^2);
    tmax = 4; tplot = .075;
    plotgap = round(Int,tplot/dt); dt = tplot/plotgap;
    nplots = round(Int,tmax/tplot);
    plotdata = [v  zeros(N+1,nplots)]; tdata = 0;
    for i = 1:nplots
        global v, vold
        global plotdata, tdata 
        for n = 1:plotgap
            w = chebfft(chebfft(v)); w[1] = 0; w[N+1] = 0;
            vnew = 2*v - vold + dt^2*w; vold = v; v = vnew;
        end
        plotdata[:,i+1] = v; tdata=[tdata;dt*i*plotgap];
    end
    
    # Plot results:
    clf(); mesh(x,tdata,plotdata',ccount=0,rcount=N+1);
    axis([-1,1,0,tmax]); zlim(-2,2); view(10,70); grid(false);
    xlabel("x"); ylabel("t"); zlabel("u");
end

function p20()
    # p20.m - 2nd-order wave eq. in 2D via FFT (compare p19.m)
    
    # Grid and initial data:
    N = 24; x = y = cheb(N)[2];
    dt = 6/N^2;
    xx = yy = x[end:-1:1];
    plotgap = round(Int,(1/3)/dt); dt = (1/3)/plotgap;
    vv = @. exp(-40*((x'-.4)^2 + y^2));
    vvold = vv;  clf();
    
    # Time-stepping by leap frog formula:
    for n = 0:3*plotgap
        t = n*dt;
        if rem(n+.5,plotgap)<1     # plots at multiples of t=1/3
              i = n÷plotgap+1;
              subplot(2,2,i,projection="3d");
              xxx = yyy = -1:1/16:1;
              s = Spline2D(xx,yy,reverse(vv,dims=:));
              vvv = evalgrid(s,xxx,yyy);
              surf(xxx,yyy,vvv); xlim(-1,1); ylim(-1,1); zlim(-0.15,1);
              view(-37.5,30); title("t = $(round(t,sigdigits=5))");
        end
        uxx = zeros(N+1,N+1); uyy = zeros(N+1,N+1);
        ii = 2:N;
        for i = 2:N                # 2nd derivs wrt x in each row
              v = vv[i,:]; V = [v;reverse(v[ii])];
              U = real(fft(V));
              W1 = real(ifft(1im*[0:N-1;0;1-N:-1].*U)); # diff wrt theta
              W2 = real(ifft(-[0:N;1-N:-1].^2 .*U));     # diff^2 wrt theta
              uxx[i,ii] = W2[ii]./(1 .-x[ii].^2) - x[ii].*W1[ii]./(1 .-x[ii].^2).^(3/2);
        end
        for j = 2:N                # 2nd derivs wrt y in each column
              v = vv[:,j]; V = [v; reverse(v[ii])];
              U = real(fft(V));
              W1 = real(ifft(1im*[0:N-1;0;1-N:-1].*U));# diff wrt theta
              W2 = real(ifft(-[0:N;1-N:-1].^2 .*U));    # diff^2 wrt theta
              uyy[ii,j] = W2[ii]./(1 .-y[ii].^2) - y[ii].*W1[ii]./(1 .-y[ii].^2).^(3/2);
        end
        vvnew = 2*vv - vvold + dt^2*(uxx+uyy);
        vvold = vv; vv = vvnew;
    end
end

function p21()
    # p21.jl - eigenvalues of Mathieu operator -u_xx + 2qcos(2x)u
    #         (compare p8.jl and p. 724 of Abramowitz & Stegun)
    
    N = 42; h = 2*pi/N; x = h*(1:N);
    D2 = toeplitz([-pi^2/(3*h^2)-1/6; @. -.5*(-1)^(1:N-1)/sin(h*(1:N-1)/2)^2]);
    qq = 0:.2:15; data = zeros(0,11);
    for q = qq;
        e = sort(eigvals(-D2 + 2*q*diagm(cos.(2*x))));
        data = [data; e[1:11]'];
    end
    clf(); subplot(1,2,1);
    plot(qq,data[:,1:2:end],"b-");
    plot(qq,data[:,2:2:end],"b--");
    xlabel("q"); ylabel("λ");
    axis([0,15,-24,32]); yticks(-24:4:32);
end

function p22()
    # p22.jl - 5th eigenvector of Airy equation u_xx = lambda*x*u
    
    clf();
    for N = 12:12:48
        D,x = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
        lam,V = eigen(D2,diagm(x[2:N]));      # generalized ev problem
        ii = findall(lam.>0);
        V = V[:,ii]; lam = lam[ii];
        ii = sortperm(lam)[5]; lambda = lam[ii];
        v = [0;V[:,ii];0]; v = v/v[Int(N/2+1)]*airyai(0);
        xx = -1:.01:1; vv = fit(x,v).(xx);
        subplot(2,2,N÷12); plot(xx,vv); grid(true)
        title("N = $N     eig = $(round(lambda,sigdigits=13))");
    end
end

function p23()
    # p23.jl - eigenvalues of perturbed Laplacian on [-1,1]x[-1,1]
    #         (compare p16.jl)
    
    # Set up tensor product Laplacian and compute 4 eigenmodes:
    N = 16; (D,x) = cheb(N); y = x;
    xx = x[2:N]; yy = y[2:N];
    D2 = D^2; D2 = D2[2:N,2:N]; 
    L = -kron(I(N-1),D2) - kron(D2,I(N-1));                #  Laplacian
    f = @. exp(20*(yy-xx'-1));      # perturbation
    L = L + diagm(f[:]);
    D,V = eigen(L); ii = sortperm(D)[1:4];
    D = D[ii]; V = V[:,ii];
    
    # Reshape them to 2D grid, interpolate to finer grid, and plot:
    xx = yy = x[end:-1:1];
    xxx = yyy = -1:.02:1;
    uu = zeros(N+1,N+1);
    ay,ax = (repeat([.56 .04],outer=(2,1)),repeat([.1,.5],outer=(1,2))); clf();
    for i = 1:4
        uu[2:N,2:N] = reshape(V[:,i],N-1,N-1);
        uu = uu/norm(uu[:],Inf);
        s = Spline2D(xx,yy,reverse(uu,dims=:));
        uuu = evalgrid(s,xxx,yyy);
        PyPlot.axes([ax[i],ay[i],.38,.38]);
        contour(xxx,yyy,uuu,levels=-.9:.2:.9);
        axis("square");
        title("eig = $(round(D[i]/(pi^2/4),sigdigits=13)) π^2/4");
    end
end

function p23a()
    # p23a.jl - eigenvalues of UNperturbed Laplacian on [-1,1]x[-1,1]
    #         (compare p16.jl)
    
    # Set up tensor product Laplacian and compute 4 eigenmodes:
    N = 16; (D,x) = cheb(N); y = x;
    xx = x[2:N]; yy = y[2:N];
    D2 = D^2; D2 = D2[2:N,2:N]; 
    L = -kron(I(N-1),D2) - kron(D2,I(N-1));                #  Laplacian
    #f = @. exp(20*(yy-xx'-1));      # perturbation
    #L = L + diagm(f[:]);
    D,V = eigen(L); ii = sortperm(D)[1:4];
    D = D[ii]; V = V[:,ii];
    
    # Reshape them to 2D grid, interpolate to finer grid, and plot:
    xx = yy = x[end:-1:1];
    xxx = yyy = -1:.02:1;
    uu = zeros(N+1,N+1);
    ay,ax = (repeat([.56 .04],outer=(2,1)),repeat([.1,.5],outer=(1,2))); clf();
    for i = 1:4
        uu[2:N,2:N] = reshape(V[:,i],N-1,N-1);
        uu = uu/norm(uu[:],Inf);
        s = Spline2D(xx,yy,reverse(uu,dims=:));
        uuu = evalgrid(s,xxx,yyy);
        PyPlot.axes([ax[i],ay[i],.38,.38]);
        contour(xxx,yyy,uuu,levels=-.9:.2:.9);
        axis("square");
        title("eig = $(round(D[i]/(pi^2/4),sigdigits=13)) π^2/4");
    end
end

function p24()
    # p24.jl - pseudospectra of Davies's complex harmonic oscillator
    #         (For finer, slower plot, change 0:2 to 0:.5.)
    
    # Eigenvalues:
    N = 70; (D,x) = cheb(N); x = x[2:N];
    L = 6; x = L*x; D = D/L;                   # rescale to [-L,L]
    A = -D^2; A = A[2:N,2:N] + (1+3im)*diagm(x.^2);
    lambda = eigvals(A);
    clf(); plot(real(lambda),imag(lambda),"k.",markersize=6);
    axis([0,50,0,40])
    
    # Pseudospectra:
    x = 0:1:50; y = 0:1:40;  zz = x' .+ 1im*y;
    minsvd(z) = minimum(svdvals(z*I-A));
    sigmin = [ minsvd(x[i]+1im*y[j]) for i=eachindex(x), j=eachindex(y) ];
    contour(x,y,sigmin',levels=10.0.^(-4:.5:-.5));
end

function p24fine()
    # p24.jl - pseudospectra of Davies's complex harmonic oscillator
    #         (For finer, slower plot, change 0:2 to 0:.5.)
    
    # Eigenvalues:
    N = 70; (D,x) = cheb(N); x = x[2:N];
    L = 6; x = L*x; D = D/L;                   # rescale to [-L,L]
    A = -D^2; A = A[2:N,2:N] + (1+3im)*diagm(x.^2);
    lambda = eigvals(A);
    clf(); plot(real(lambda),imag(lambda),"k.",markersize=6);
    axis([0,50,0,40]);
    
    # Pseudospectra:
    x = 0:.5:50; y = 0:.5:40;
    minsvd(z) = minimum(svdvals(z*I-A));
    sigmin = [ minsvd(x[i]+1im*y[j]) for i=eachindex(x), j=eachindex(y) ];
    contour(x,y,sigmin',levels=10.0.^(-4:.5:-.5));
end

function p25()
    # p25.jl - stability regions for ODE formulas
    
    # Adams-Bashforth:
    clf(); subplot(221);
    zplot(z) = plot(real(z),imag(z));
    plot([-8,8],[0,0],"k-"); plot([0,0],[-8,8],"k-");
    z = exp.(1im*pi*(0:200)/100); r = z.-1;
    s = 1; zplot(r./s);                                  # order 1
    s = @. (3-1/z)/2; zplot(r./s);                         # order 2
    s = @. (23-16/z+5/z^2)/12; zplot(r./s);             # order 3
    axis("square"); axis([-2.5,.5,-1.5,1.5]); grid(true)
    title("Adams—Bashforth");
    
    # Adams-Moulton:
    subplot(222);
    plot([-8,8],[0,0],"k-"); plot([0,0],[-8,8],"k-");
    s = @. (5*z+8-1/z)/12; zplot(r./s);                    # order 3
    s = @. (9*z+19-5/z+1/z^2)/24; zplot(r./s);           # order 4
    s = @. (251*z+646-264/z+106/z^2-19/z^3)/720; zplot(r./s) ;   # 5
    d = @. 1-1/z;
    s = @. 1-d/2-d^2/12-d^3/24-19*d^4/720-3*d^5/160; zplot(d./s); # 6
    axis("square"); axis([-7,1,-4,4]); grid(true);
    title("Adams—Moulton");
    
    # Backward differentiation:
    subplot(223);
    plot([-40,40],[0,0],"k-"); plot([0,0],[-40,40],"k-");
    r = zeros(size(d))
    for i = 1:6  # orders 1-6
        r += @. (d^i)/i;
        zplot(r);
    end
    axis("square"); axis([-15,35,-25,25]); grid(true);
    title("backward differentiation")
    
    # Runge-Kutta:
    subplot(224);
    plot([-8,8],[0,0],"k-"); plot([0,0],[-8,8],"k-");
    w = complex(zeros(4)); W = transpose(w);
    for i = 2:length(z)
        # orders 1-4
        w[1] -= (1+w[1]-z[i]);
        w[2] -= (1+w[2]+.5*w[2]^2-z[i]^2)/(1+w[2]);
        w[3] -= (1+w[3]+.5*w[3]^2+w[3]^3/6-z[i]^3)/(1+w[3]+w[3]^2/2);
        w[4] -= (1+w[4]+.5*w[4]^2+w[4]^3/6+w[4].^4/24-z[i]^4)/(1+w[4]+w[4]^2/2+w[4]^3/6);
        W = [W; transpose(w)];
    end
    zplot(W); axis("square"); axis([-5,2,-3.5,3.5]); grid(true);
    title("Runge—Kutta");
end

function p26()
    # p26.m - eigenvalues of 2nd-order Chebyshev diff. matrix
    
    N = 60; (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
    (lam,V) = eigen(D2);
    ii = sortperm(-lam); e = lam[ii]; V = V[:,ii];
    
    # Plot eigenvalues:
    clf(); PyPlot.axes([.1, .62, .8, .3]);
    loglog(-e,".",markersize=4); ylabel("eigenvalue");
    title("N = $N       max |λ| = $(round(maximum(-e)/N^4,sigdigits=5)) \$N^4\$");
    semilogy(2*N/pi*[1,1],[1,1e6],"--r");
    text(2.1*N/pi,24,"2π / N",fontsize=12);
    
    # Plot eigenmodes N/4 (physical) and N (nonphysical):
    vN4 = [0; V[:,Int(N/4-1)]; 0];
    xx = -1:.01:1; vv = fit(x,vN4).(xx);
    PyPlot.axes([.1, .36, .8, .15]); plot(xx,vv);
    plot(x,vN4,"k.",markersize=4); title("eigenmode N/4");
    vN = V[:,N-1];
    PyPlot.axes([.1, .1, .8, .15]);
    semilogy(x[2:N],abs.(vN)); axis([-1,1,5e-6,1]);
    plot(x[2:N],abs.(vN),"k.",markersize=4);
    title("absolute value of eigenmode N    (log scale)");
end

function p27()
    # p27.jl - Solve KdV eq. u_t + uu_x + u_xxx = 0 on [-pi,pi] by
    #         FFT with integrating factor v = exp(-ik^3t)*u-hat.
    
    # Set up grid and two-soliton initial data:
    N = 256; dt = .4/N^2; x = (2*pi/N)*(-N/2:N/2-1);
    A = 25; B = 16; clf();
    u = @. 3*A^2*sech(.5*(A*(x+2)))^2 + 3*B^2*sech(.5*(B*(x+1)))^2;
    v = fft(u); k = [0:N/2-1;0;-N/2+1:-1]; ik3 = 1im*k.^3;
    
    # Solve PDE and plot results:
    tmax = 0.006; nplt = floor(Int,(tmax/25)/dt); nmax = round(Int,tmax/dt);
    udata = u; tdata = [0.0];
    for n = 1:nmax
        t = n*dt; g = -.5im*dt*k;
        E = exp.(dt*ik3/2); E2 = E.^2;
        a = g.*fft(real( ifft(     v    ) ).^2);
        b = g.*fft(real( ifft(E.*(v+a/2)) ).^2);     # 4th-order
        c = g.*fft(real( ifft(E.*v + b/2) ).^2);     # Runge-Kutta
        d = g.*fft(real( ifft(E2.*v+E.*c) ).^2);
        v = E2.*v + (E2.*a + 2*E.*(b+c) + d)/6;
        if mod(n,nplt) == 0
            u = real(ifft(v));
            udata = [udata u]; tdata = [tdata;t];
        end
    end
    mesh(x,tdata,udata',ccount=0,rcount=N); view(-20,25);
    xlabel("x"); ylabel("y"); grid(true);
    xlim(-pi,pi); ylim(0,tmax); zlim(0,12000);|
    zticks([0,2000]);
end

function p28()
    # p28.jl - eigenmodes of Laplacian on the disk (compare p22.jl)
    
    # r coordinate, ranging from -1 to 1 (N must be odd):
    N = 25; N2 = Int((N-1)/2);
    (D,r) = cheb(N); D2 = D^2;
    D1 = D2[2:N2+1,2:N2+1]; D2 = D2[2:N2+1,N:-1:N2+2];
    E1 =  D[2:N2+1,2:N2+1]; E2 =  D[2:N2+1,N:-1:N2+2];
    
    # t = theta coordinate, ranging from 0 to 2*pi (M must be even):
    M = 20; dt = 2*pi/M; t = dt*(1:M); M2 = Int(M/2);
    D2t = toeplitz([-pi^2/(3*dt^2)-1/6; @. .5*(-1)^(2:M)/sin(dt*(1:M-1)/2)^2]);
    
    # Laplacian in polar coordinates:
    R = diagm(1 ./r[2:N2+1]);
    Z = zeros(M2,M2); 
    L = kron(D1+R*E1,I(M)) + kron(D2+R*E2,[Z I(M2);I(M2) Z]) + kron(R^2,D2t);
    
    # Compute four eigenmodes:
    index = [1,3,6,10];
    Lam,V = eigen(-L); ii = sortperm(abs.(Lam))[index];
    Lam = Lam[ii]; V = V[:,ii];
    Lam = sqrt.(real(Lam/Lam[1]));
    
    # Plot eigenmodes with nodal lines underneath:
    (rr,tt) = (r[1:N2+1],[0;t]);
    (xx,yy) = @. (cos(tt)*rr',sin(tt)*rr');
    z = exp.(1im*pi*(-100:100)/100);
    for i = 1:4
        figure(i); clf();
        u = reshape(real(V[:,i]),M,N2);
        u = [zeros(M+1) u[[M;1:M],:]];
        u = u/norm(u[:],Inf);
        #plot3D(real(z),imag(z),zeros(size(z)));
        xlim(-1.05,1.05); ylim(-1.05,1.05); zlim(-1.05,1.05); axis("off");
        surf(xx,yy,u)
        contour3D(xx,yy,u.-1,levels=[-1]);
        plot3D(real(z),imag(z),-abs.(z));
        title("Mode $(index[i]):  λ = $(round(Lam[i],sigdigits=11))",fontsize=9);
    end
end

function p28b()
    # p28b.jl - eigenmodes of Laplacian on the disk
    
    # r coordinate, ranging from -1 to 1 (N must be odd)
    N = 25; N2 = Int((N-1)/2);
    (D,r) = cheb(N); D2 = D^2;
    D1 = D2[2:N2+1,2:N2+1]; D2 = D2[2:N2+1,N:-1:N2+2];
    E1 =  D[2:N2+1,2:N2+1]; E2 =  D[2:N2+1,N:-1:N2+2];
    
    # t = theta coordinate, ranging from 0 to 2*pi (M must be even):
    M = 20; dt = 2*pi/M; t = dt*(1:M); M2 = Int(M/2);
    D2t = toeplitz([-pi^2/(3*dt^2)-1/6; @. .5*(-1)^(2:M)/sin(dt*(1:M-1)/2)^2]);
    
    # Laplacian in polar coordinates:
    R = diagm(1 ./r[2:N2+1]);
    Z = zeros(M2,M2); 
    L = kron(D1+R*E1,I(M)) + kron(D2+R*E2,[Z I(M2);I(M2) Z]) + kron(R^2,D2t);
    
    # Compute 25 eigenmodes:
    index = 1:25;
    Lam,V = eigen(-L); ii = sortperm(abs.(Lam))[index];
    Lam = Lam[ii]; V = V[:,ii];
    Lam = sqrt.(real(Lam/Lam[1]));
    
    # Plot nodal lines:
    (rr,tt) = (r[1:N2+1],[0;t]);
    (xx,yy) = @. (cos(tt)*rr',sin(tt)*rr');
    z = exp.(1im*pi*(-100:100)/100);  clf();
    for i = 1:25
        subplot(5,5,i);
        u = reshape(real(V[:,i]),M,N2);
        u = [zeros(M+1) u[[M;1:M],:]];
        u = u/norm(u[:],Inf);
        plot(real(z),imag(z));
        xlim(-1.07,1.07); ylim(-1.07,1.07); axis("off"); axis("equal");
        contour(xx,yy,u,levels=[0]);
        title("$(round(Lam[i],sigdigits=5))",fontsize=8);
    end
end

function p29()
    # p29.jl - solve Poisson equation on the unit disk
    #         (compare p16.jl and p28.jl)
    
    # Laplacian in polar coordinates:
    N = 25; N2 = Int((N-1)/2);
    (D,r) = cheb(N); D2 = D^2;
    D1 = D2[2:N2+1,2:N2+1]; D2 = D2[2:N2+1,N:-1:N2+2];
    E1 =  D[2:N2+1,2:N2+1]; E2 =  D[2:N2+1,N:-1:N2+2];
    M = 20; dt = 2*pi/M; t = dt*(1:M); M2 = Int(M/2);
    D2t = toeplitz([-pi^2/(3*dt^2)-1/6; @. .5*(-1)^(2:M)/sin(dt*(1:M-1)/2)^2]);
    R = diagm(1 ./r[2:N2+1]); Z = zeros(M2,M2);
    L = kron(D1+R*E1,I(M)) + kron(D2+R*E2,[Z I(M2);I(M2) Z]) + kron(R^2,D2t);
    
    # Right-hand side and solution for u:
    (rr,tt) = (r[2:N2+1]',t);
    f = @. -rr^2*sin(tt/2)^4 + sin(6*tt)*cos(tt/2)^2; u = L\f[:];
    
    # Reshape results onto 2D grid and plot them:
    u = reshape(u,M,N2); u = [zeros(M+1) u[[M;1:M],:]];
    (rr,tt) = (r[1:N2+1],t[[M;1:M]]);
    (xx,yy) = @. (cos(tt)*rr',sin(tt)*rr');
    clf();
    surf(xx,yy,u);
    gca()[:view_init](40,-70);
    xlim(-1,1); ylim(-1,1); zlim(-.01,.05);
    xlabel("x"); ylabel("y"); zlabel("z");
end

function p30()
    # p30.jl - spectral integration, ODE style (compare p12.jl)
    
    # Computation: various values of N, four functions:
    Nmax = 50; E = zeros(4,Nmax); clf();
    for N = 1:Nmax
        i = 1:N; (D,x) = cheb(N);
        x = x[i]; Di = inv(D[i,i]); w = Di[1,:];
        f = @. abs(x)^3;     E[1,N] = abs(dot(w,f) - .5);
        f = @. exp(-x^(-2)); E[2,N] = abs(dot(w,f) - 2*(exp(-1)+sqrt(pi)*(erf(1)-1)));
        f = @. 1/(1+x^2);    E[3,N] = abs(dot(w,f) - pi/2);
        f = x.^10;           E[4,N] = abs(dot(w,f) - 2/11);
    end
    
    # Plot results:
    labels = [L"|x|^3",L"\exp(-x^2)",L"1/(1+x^2)",L"x^{10}"];
    for iplot = 1:4
        subplot(2,2,iplot)
        semilogy(E[iplot,:].+1e-100,".-",markersize=10);
        axis([0,Nmax,1e-18,1e3]); grid(true);
        xticks(0:10:Nmax); yticks((10.0).^(-15:5:0));
        ylabel("error"); text(32,.004,labels[iplot]);
    end
end

function p30b()
    # p30b.jl - spectral integration, Clenshaw-Curtis style (compare p30.jl)
    
    # Computation: various values of N, four functions:
    Nmax = 50; E = zeros(4,Nmax); clf();
    for N = 1:Nmax
        (x,w) = clencurt(N);
        f = @. abs(x)^3;     E[1,N] = abs(dot(w,f) - .5);
        f = @. exp(-x^(-2)); E[2,N] = abs(dot(w,f) - 2*(exp(-1)+sqrt(pi)*(erf(1)-1)));
        f = @. 1/(1+x^2);    E[3,N] = abs(dot(w,f) - pi/2);
        f = x.^10;           E[4,N] = abs(dot(w,f) - 2/11);
    end
    
    # Plot results:
    labels = [L"|x|^3",L"\exp(-x^2)",L"1/(1+x^2)",L"x^{10}"];
    for iplot = 1:4
        subplot(2,2,iplot)
        semilogy(E[iplot,:].+1e-100,".-",markersize=10);
        axis([0,Nmax,1e-18,1e3]); grid(true);
        xticks(0:10:Nmax); yticks((10.0).^(-15:5:0));
        ylabel("error"); text(32,.004,labels[iplot]);
    end
end

function p30c()
    # p30c.jl - spectral integration, Gauss style (compare p30.jl)
    
    # Computation: various values of N, four functions:
    Nmax = 50; E = zeros(4,Nmax); clf();
    for N = 1:Nmax
        (x,w) = gauss(N);
        f = @. abs(x)^3;     E[1,N] = abs(dot(w,f) - .5);
        f = @. exp(-x^(-2)); E[2,N] = abs(dot(w,f) - 2*(exp(-1)+sqrt(pi)*(erf(1)-1)));
        f = @. 1/(1+x^2);    E[3,N] = abs(dot(w,f) - pi/2);
        f = x.^10;           E[4,N] = abs(dot(w,f) - 2/11);
    end
    
    # Plot results:
    labels = [L"|x|^3",L"\exp(-x^2)",L"1/(1+x^2)",L"x^{10}"];
    for iplot = 1:4
        subplot(2,2,iplot)
        semilogy(E[iplot,:].+1e-100,".-",markersize=10);
        axis([0,Nmax,1e-18,1e3]); grid(true);
        xticks(0:10:Nmax); yticks((10.0).^(-15:5:0));
        ylabel("error"); text(32,.004,labels[iplot]);
    end
end

function p31()
    # p31.jl - gamma function via complex integral, trapezoid rule
    
    N = 70; theta = @. -pi + (2*pi/N)*(.5:N-.5);
    c = -11;                     # center of circle of integration
    r = 16;                      # radius of circle of integration
    x = -3.5:.1:4; y = -2.5:.1:2.5;
    zz = x' .+ 1im*y; gaminv = 0*zz;
    for i = 1:N
        t = c + r*exp(1im*theta[i]);
        gaminv += exp(t)*t.^(-zz)*(t-c);
    end
    gaminv = gaminv/N; gam = 1 ./gaminv;
    clf(); surf(x,y,abs.(gam));
    xlim(-3.5,4); ylim(-2.5,2.5); zlim(0,6);
    xlabel("Re(z)"); ylabel("Im(z)");
    text3D(4,-1.4,5.5,"\$|\\Gamma(z)|\$",fontsize=20);
end

function p32()
    # p32.jl - solve u_xx = exp(4x), u(-1)=0, u(1)=1 (compare p13.jl)
    
    N = 16;
    (D,x) = cheb(N);
    D2 = D^2;
    D2 = D2[2:N,2:N];                   # boundary conditions
    f = @. exp(4*x[2:N]);
    u = D2\f;                           # Poisson eq. solved here
    u = [0;u;0] + (x.+1)/2;
    clf();
    plot(x,u,".",markersize=10);
    xx = -1:.01:1;
    uu = fit(x,u).(xx);
    plot(xx,uu); grid(true);
    exact = @. (exp(4*xx) - sinh(4)*xx - cosh(4))/16 + (xx+1)/2;
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=4))",fontsize=12);
end

function p33()
    # p33.jl - solve linear BVP u_xx = exp(4x), u'(-1)=u(1)=0
    
    N = 16;
    (D,x) = cheb(N);
    D2 = D^2;
    D2[N+1,:] = D[N+1,:];            # Neumann condition at x = -1
    D2 = D2[2:N+1,2:N+1];
    f = @. exp(4*x[2:N]);
    u = D2\[f;0];
    u = [0;u];
    clf();
    plot(x,u,".",markersize=10);
    axis([-1,1,-4,0]);
    xx = -1:.01:1;
    uu = fit(x,u).(xx);
    plot(xx,uu); grid(true);
    exact = @. (exp(4*xx) - 4*exp(-4)*(xx-1) - exp(4))/16;
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=5))",fontsize=12);
end

function p34()
    # p34.jl - Allen-Cahn eq. u_t = eps*u_xx+u-u^3, u(-1)=-1, u(1)=1
    #         (compare p6.jl and p32.jl)
    
    # Differentiation matrix and initial data:
    N = 20; (D,x) = cheb(N); D2 = D^2;     # use full-size matrix
    D2[[1,N+1],:] .= 0;                     # for convenience
    eps = 0.01; dt = min(.01,50/(N^4*eps));
    t = 0.0; v = @. .53*x + .47*sin(-1.5*pi*x);
    
    # Solve PDE by Euler formula and plot results:
    tmax = 100; tplot = 2; nplots = round(Int,tmax/tplot);
    plotgap = round(Int,tplot/dt); dt = tplot/plotgap;
    xx = -1:.025:1; vv = fit(x,v).(xx);
    plotdata = [vv zeros(length(xx),nplots)]; tdata = t;
    for i = 1:nplots
        for n = 1:plotgap
            t = t+dt; v = v + dt*(eps*D2*(v-x) + v - v.^3);    # Euler
        end
        vv = fit(x,v).(xx);
        plotdata[:,i+1] = vv; tdata = [tdata; t];
    end
    clf();
    surf(xx,tdata,plotdata'); grid(true);
    xlim(-1,1); ylim(0,tmax); zlim(-1,1);
    gca()[:view_init](55,-150);
    xlabel("x"); ylabel("t"); zlabel("u");
end

function p35()
    # p35.m - Allen-Cahn eq. as in p34.m, but with boundary condition
    #         imposed explicitly ("method (II)")
    
    # Differentiation matrix and initial data:
    N = 20; (D,x) = cheb(N); D2 = D^2;     # use full-size matrix
    eps = 0.01; dt = min(.01,50/(N^4*eps));
    t = 0.0; v = @. .53*x + .47*sin(-1.5*pi*x);
    
    # Solve PDE by Euler formula and plot results:
    tmax = 100; tplot = 2; nplots = round(Int,tmax/tplot);
    plotgap = round(Int,tplot/dt); dt = tplot/plotgap;
    xx = -1:.025:1; vv = fit(x,v).(xx);
    plotdata = [vv zeros(length(xx),nplots)]; tdata = t;
    for i = 1:nplots
        for n = 1:plotgap
            t = t+dt; v = v + dt*(eps*D2*(v-x) + v - v.^3);    # Euler
            v[1] = 1 + sin(t/5)^2;  v[end] = -1;               # BC
        end
        vv = fit(x,v).(xx);
        plotdata[:,i+1] = vv; tdata = [tdata; t];
    end
    clf();
    surf(xx,tdata,plotdata'); grid(true);
    xlim(-1,1); ylim(0,tmax); zlim(-1,2);
    gca()[:view_init](55,-150);
    xlabel("x"); ylabel("t"); zlabel("u");
end

function p36()
    # p36.m - Laplace eq. on [-1,1]x[-1,1] with nonzero BCs
    
    # Set up grid and 2D Laplacian, boundary points included:
    N = 24; (D,x) = cheb(N); y = x;
    xx = repeat(x',outer=(N+1,1)); yy = repeat(y,outer=(1,N+1));
    D2 = D^2; L = kron(I(N+1),D2) + kron(D2,I(N+1));
    
    # Impose boundary conditions by replacing appropriate rows of L:
    b = @. (abs(xx[:])==1) | (abs(yy[:])==1);            # boundary pts
    L[b,:] .= 0; L[b,b] = I(4*N);
    rhs = zeros((N+1)^2);
    rhs[b] = @. (yy[b]==1)*(xx[b]<0)*sin(pi*xx[b])^4 + .2*(xx[b]==1)*sin(3*pi*yy[b]);
    
    # Solve Laplace equation, reshape to 2D, and plot:
    u = L\rhs; uu = reshape(u,N+1,N+1);
    xxx = yyy = -1:.04:1;
    s = Spline2D(x[end:-1:1],y[end:-1:1],reverse(uu,dims=:),kx=1,ky=1);
    clf();
    surf(xxx,yyy,s.(xxx,yyy'));
    xlim(-1,1); ylim(-1,1); zlim(-.2,1);
    gca()[:view_init](45,-110);
    umid = uu[Int(N/2)+1,Int(N/2)+1];
    text3D(0,.8,.4,"u(0,0) = $(round(umid,sigdigits=9))");
end

function p37()
    # p37.jl - 2D "wave tank" with Neumann BCs for |y|=1
    
    # x variable in [-A,A], Fourier:
    A = 3; Nx = 50; dx = 2*A/Nx; x = @. -A+dx*(1:Nx);
    D2x = (pi/A)^2*toeplitz([-1/(3*(dx/A)^2)-1/6;
            @. .5*(-1).^(2:Nx)/sin((pi*dx/A)*(1:Nx-1)/2)^2]);
    
    # y variable in [-1,1], Chebyshev:
    Ny = 15; Dy,y = cheb(Ny); D2y = Dy^2;
    BC = -Dy[[1,Ny+1],[1,Ny+1]]\Dy[[1,Ny+1],2:Ny];
    
    # Grid and initial data:
    vv = @. exp(-8*((x'+1.5)^2+y^2));
    dt = 5/(Nx+Ny^2);
    vvold = @. exp(-8*((x'+dt+1.5).^2+y.^2));
    
    # Time-stepping by leap frog formula:
    plotgap = round(Int,2/dt); dt = 2/plotgap;
    for n = 0:2*plotgap
        t = n*dt;
        if rem(n+.5,plotgap)<1
            figure(n/plotgap+1); clf(); surf(x,y,vv);
            gca()[:view_init](60,-100);
            xlim(-A,A); ylim(-1,1); zlim(-0.15,1);
            text3D(-2.5,1,.5,"t = $(round(t))",fontsize=18);
            zticks([]);
        end
        vvnew = 2*vv - vvold + dt^2*(vv*D2x +D2y*vv);
        vvold = vv; vv = vvnew;
        vv[[1,Ny+1],:] = BC*vv[2:Ny,:];       # Neumann BCs for |y|=1
    end
end

function p38()
    # p38.jl - solve u_xxxx = exp(x), u(-1)=u(1)=u'(-1)=u'(1)=0
    #         (compare p13.jl)
    
    # Construct discrete biharmonic operator:
    N = 15; (D,x) = cheb(N);
    S = diagm([0; 1 ./(1 .-x[2:N].^2); 0]);
    D4 = (diagm(1 .-x.^2)*D^4 - 8*diagm(x)*D^3 - 12*D^2)*S;
    D4 = D4[2:N,2:N];
    
    # Solve boundary-value problem and plot result:
    f = @. exp(x[2:N]); u = D4\f; u = [0;u;0];
    clf(); axes([.1,.4,.8,.5]);
    plot(x,u,".",markersize=10);
    axis([-1,1,-.01,.06]); grid(true);
    xx = (-1:.01:1);
    uu = (1 .-xx.^2).*fit(x,S*u).(xx);
    plot(xx,uu);
    
    # Determine exact solution and print maximum error:
    A = [1 -1 1 -1; 0 1 -2 3; 1 1 1 1; 0 1 2 3];
    V = xx.^(0:3)';
    c = A\exp.([-1,-1,1,1]); exact = exp.(xx) - V*c;
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=5))",fontsize=12);
end

function p39()
    # p39.m - eigenmodes of biharmonic on a square with clamped BCs
    #         (compare p38.m)
    
    # Construct spectral approximation to biharmonic operator:
    N = 17; (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
    S = diagm([0; 1 ./(1 .-x[2:N].^2); 0]);
    D4 = (diagm(1 .-x.^2)*D^4 - 8*diagm(x)*D^3 - 12*D^2)*S;
    D4 = D4[2:N,2:N]; 
    L = kron(I(N-1),D4) + kron(D4,I(N-1)) + 2*kron(D2,I(N-1))*kron(I(N-1),D2);
    
    # Find and plot 25 eigenmodes:
    Lam,V = eigen(-L); Lam = -real(Lam);
    ii = sortperm(Lam)[1:25]; Lam = Lam[ii]; V = real(V[:,ii]);
    Lam = sqrt.(Lam/Lam[1]);
    y = x; xxx = yyy = -1:.01:1;
    sq = [1+1im,-1+1im,-1-1im,1-1im,1+1im]; clf();
    for i = 1:25
        uu = zeros(N+1,N+1); uu[2:N,2:N] = reshape(V[:,i],N-1,N-1);
        subplot(5,5,i); plot(real(sq),imag(sq));
        s = Spline2D(x[end:-1:1],y[end:-1:1],reverse(uu,dims=:),kx=1,ky=1);
        contour(xxx,yyy,s.(xxx,yyy'),levels=[0],color="k"); axis("square");
        axis(1.25*[-1,1,-1,1]); axis("off");
        text(-.3,1.15,"$(round(Lam[i],sigdigits=5))",fontsize=7);
    end
end

function p40()
    # p40.jl - eigenvalues of Orr-Sommerfeld operator (compare p38.jl)
    
    R = 5772; clf();
    for N = 40:20:100
        # 2nd- and 4th-order differentiation matrices:
        (D,x) = cheb(N); D2 = D^2; D2 = D2[2:N,2:N];
        S = diagm([0; 1 ./(1 .-x[2:N].^2); 0]);
        D4 = (diagm(1 .-x.^2)*D^4 - 8*diagm(x)*D^3 - 12*D^2)*S;
        D4 = D4[2:N,2:N];
    
        # Orr-Sommerfeld operators A,B and generalized eigenvalues:
        A = (D4-2*D2+I)/R - 2im*I - 1im*diagm(1 .-x[2:N].^2)*(D2-I);
        B = D2-I;
        ee = eigvals(A,B);
        i = N÷20-1; subplot(2,2,i);
        plot(real(ee),imag(ee),".",markersize=8);
        grid(true); axis("square"); axis([-.8,.2,-1,0]);
        title("N = $N    \$\\lambda_{max}\$ = $(round(maximum(real(ee)),sigdigits=7))");
    end
end
