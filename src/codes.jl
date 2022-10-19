# p1 - convergence of fourth-order finite differences
function p1()
    # For various N, set up grid in [-π,π] and function u(x):
    Nvec = @. 2^(3:12)
    clf()
    PyPlot.axes([0.1, 0.4, 0.8, 0.5])
    for N = Nvec
        h = 2π / N
        x = @. -π + (1:N) * h
        u = @. exp(sin(x)^2)
        uprime = @. 2sin(x) * cos(x) * u

        # Construct sparse fourth-order differentiation matrix:
        col1 = [ 0; -2/3h; 1/12h; zeros(N-5); -1/12h; 2/3h ]
        D = sparse( [col1[mod(i-j,N) + 1] for i in 1:N, j in 1:N] )

        # Plot max(abs(D*u-uprime)):
        error = norm(D * u - uprime, Inf)
        loglog(N, error, "k.", markersize=6)
    end
    grid(true)
    xlabel("N")
    ylabel("error")
    title("Convergence of fourth-order finite differences")
    loglog(Nvec, 1.0 ./ Nvec .^ 4, "--")
    text(105, 5e-8, L"N^{-4}", fontsize=14)
    return gcf()
end

# p2 - convergence of periodic spectral method (compare p1.jl)
function p2()
    # For various N (even), set up grid as before:
    clf()
    for N = 2:2:100
        h = 2π / N
        x = [-π + i * h for i = 1:N]
        u = @. exp(sin(x))
        uprime = @. cos(x) * u

        # Construct spectral differentiation matrix:
        entry(k) = k==0 ? 0 : (-1)^k * 0.5cot( k * h / 2 )
        D = [ entry(mod(i-j,N)) for i in 1:N, j in 1:N ]

        # Plot max(abs(D*u-uprime)):
        error = norm(D * u - uprime, Inf)
        loglog(N, error, "k.", markersize=6)
    end
    grid(true)
    xlabel("N")
    ylabel("error")
    title("Convergence of spectral differentiation")
    return gcf()
end

# p3 - band-limited interpolation
function p3()
    h = 1
    xmax = 10
    clf()
    x = -xmax:h:xmax                     # computational grid
    xx = -xmax-h/20:h/10:xmax+h/20       # plotting grid
    v = zeros(length(x), 3)
    v[:, 1] = @. float(x == 0)
    v[:, 2] = @. float(abs(x) ≤ 3)
    v[:, 3] = @. max(0, 1 - abs(x) / 3)
    for plt = 1:3
        subplot(4, 1, plt)
        plot(x, v[:, plt], ".", markersize=6)
        p = 0
        for i = 1:length(x)
            p = @. p + v[i, plt] * sinpi((xx - x[i]) / h) / (π * (xx - x[i]) / h)
        end
        plot(xx, p, "-")
        axis([-xmax, xmax, -0.5, 1.5])
        xticks(1:0)
        yticks(0:1)
    end
    return gcf()
end

# p3 - band-limited interpolation
function p3g()
    xmax = 6
    clf()
    for (plt,h) in enumerate([1,1/2,1/10])
        x = -xmax:h:xmax                     # computational grid
        xx = -xmax-h/20:h/10:xmax+h/20       # plotting grid
        v = @. float(abs(x) ≤ 3)
        subplot(3, 1, plt)
        plot(x, v, ".", markersize=6)
        # p = zero(xx)
        # for i in eachindex(x)
            # @. p += v[i] * sinpi((xx - x[i]) / h) / (π * (xx - x[i]) / h)
        # end
        BLI(t) = sum(v[i] * sinpi((t - x[i]) / h) / (π * (t - x[i]) / h) for i in eachindex(x)) 
        plot(xx, BLI.(xx), "-")
        axis([-xmax, xmax, -0.25, 1.25])
        xticks(1:0)
        yticks(0:1)
    end
    return gcf()
end

# p4 - periodic spectral differentiation
function p4()
    # Set up grid and differentiation matrix:
    N = 24
    h = 2π / N
    x = h * (1:N)
    column = [0; @. 0.5 * (-1)^(1:N-1) * cot((1:N-1) * h / 2)]
    D = toeplitz(column, column[[1; N:-1:2]])

    # Differentiation of a hat function:
    v = @. max(0, 1 - abs(x - π) / 2)
    clf()
    subplot(2, 2, 1)
    plot(x, v, ".-", markersize=6)
    axis([0, 2π, -0.5, 1.5])
    grid(true)
    title("function")
    subplot(2, 2, 2), plot(x, D * v, ".-", markersize=6)
    axis([0, 2π, -1, 1])
    grid(true)
    title("spectral derivative")

    # Differentiation of exp(sin(x)):
    v = @. exp(sin(x))
    vʹ = @. cos(x) * v
    subplot(2, 2, 3), plot(x, v, ".-", markersize=6)
    axis([0, 2π, 0, 3]), grid(true)
    subplot(2, 2, 4), plot(x, D * v, ".-", markersize=6)
    axis([0, 2π, -2, 2]), grid(true)
    error = round(norm(D * v - vʹ, Inf), sigdigits=5)
    text(2.2, 1.4, "max error = $error", fontsize=8)
    return gcf()
end

# p5 - repetition of p4 via FFT
function p5()
    #        For complex v, delete "real" commands.
    # Differentiation of a hat function:
    N = 24
    h = 2π / N
    x = h * (1:N)
    v = @. max(0, 1 - abs(x - π) / 2)
    v̂ = fft(v)
    ŵ = 1im * [0:N/2-1; 0; -N/2+1:-1] .* v̂
    w = real(ifft(ŵ))
    clf()
    subplot(2, 2, 1)
    plot(x, v, ".-", markersize=6)
    axis([0, 2π, -0.5, 1.5])
    grid(true)
    title("function")
    subplot(2, 2, 2), plot(x, w, ".-", markersize=6)
    axis([0, 2π, -1, 1])
    grid(true)
    title("spectral derivative")

    # Differentiation of exp(sin(x)):
    v = @. exp(sin(x))
    vʹ = @. cos(x) * v
    v̂ = fft(v)
    ŵ = 1im * [0:N/2-1; 0; -N/2+1:-1] .* v̂
    w = real(ifft(ŵ))
    subplot(2, 2, 3), plot(x, v, ".-", markersize=6)
    axis([0, 2π, 0, 3]), grid(true)
    subplot(2, 2, 4), plot(x, w, ".-", markersize=6)
    axis([0, 2π, -2, 2]), grid(true)
    error = round(norm(w - vʹ, Inf), sigdigits=4)
    text(2.2, 1.4, "max error = $error", fontsize=8)
    return gcf()
end

function p5r()
    #        For complex v, delete "real" commands.
    # Differentiation of a hat function:
    N = 24
    h = 2π / N
    x = h * (1:N)
    v = @. max(0, 1 - abs(x - π) / 2)
    v̂ = rfft(v)
    ŵ = 1im * [0:N/2-1; 0] .* v̂
    w = irfft(ŵ,N)
    clf()
    subplot(2, 2, 1)
    plot(x, v, ".-", markersize=6)
    axis([0, 2π, -0.5, 1.5])
    grid(true)
    title("function")
    subplot(2, 2, 2), plot(x, w, ".-", markersize=6)
    axis([0, 2π, -1, 1])
    grid(true)
    title("spectral derivative")

    # Differentiation of exp(sin(x)):
    v = @. exp(sin(x))
    vʹ = @. cos(x) * v
    v̂ = rfft(v)
    ŵ = 1im * [0:N/2-1; 0] .* v̂
    w = irfft(ŵ,N)
    subplot(2, 2, 3), plot(x, v, ".-", markersize=6)
    axis([0, 2π, 0, 3]), grid(true)
    subplot(2, 2, 4), plot(x, w, ".-", markersize=6)
    axis([0, 2π, -2, 2]), grid(true)
    error = round(norm(w - vʹ, Inf), sigdigits=4)
    text(2.2, 1.4, "max error = $error", fontsize=8)
    return gcf()
end

# p6 - variable coefficient wave equation
function p6()
    # Grid, variable coefficient, and initial data:
    N = 128
    h = 2π / N
    x = h * (1:N)
    t = 0
    Δt = h / 4
    c = @. 0.2 + sin(x - 1)^2
    v = @. exp(-100 * (x - 1) .^ 2)
    vold = @. exp(-100 * (x - 0.2Δt - 1) .^ 2)

    # Time-stepping by leap frog formula:
    tmax = 8
    tplot = 0.15
    clf()
    plotgap = round(tplot / Δt)
    Δt = tplot / plotgap
    nplots = round(Int, tmax / tplot)
    data = [v zeros(N, nplots)]
    tdata = [t]
    for i in 1:nplots
        for n in 1:plotgap
            t = t + Δt
            v̂ = rfft(v)
            ŵ = 1im * [0:N/2-1; 0] .* v̂
            w = irfft(ŵ,N)
            vnew = vold - 2Δt * c .* w
            vold = v
            v = vnew
        end
        data[:, i+1] = v
        tdata = [tdata; t]
    end
    mesh(x, tdata, data', ccount=0)
    view(10, 70)
    xlim(0, 2π)
    ylim(0, tmax)
    zlim(0, 5)
    xlabel("x")
    ylabel("t")
    zlabel("u")
    return gcf()
end

# p6u - variable coefficient wave equation - UNSTABLE VARIANT
function p6u()
    # Grid, variable coefficient, and initial data:
    N = 128
    h = 2π / N
    x = h * (1:N)
    c = @. 0.2 + sin(x - 1)^2
    t = 0
    Δt = 1.9 / N
    v = @. exp(-100 * (x - 1)^2)
    vold = @. exp(-100 * (x - 0.2Δt - 1)^2)

    # Time-stepping by leap frog formula:
    tmax = 8
    tplot = 0.15
    clf()
    plotgap = round(Int, tplot / Δt)
    nplots = round(Int, tmax / tplot)
    data = [v zeros(N, nplots)]
    tdata = t
    for i = 1:nplots
        for n = 1:plotgap
            t = t + Δt
            v̂ = fft(v)
            ŵ = 1im * [0:N/2-1; 0; -N/2+1:-1] .* v̂
            w = real.(ifft(ŵ))
            vnew = vold - 2Δt * c .* w        # leap frog formula
            vold = v
            v = vnew
        end
        data[:, i+1] = v
        tdata = [tdata; t]
        if norm(v, Inf) > 2.5
            data = data[:, 1:i+1]
            break
        end
    end

    # Plot results:
    mesh(x, tdata, data', ccount=0)
    xlim(0, 2π)
    ylim(0, tmax)
    zlim(-3, 3)
    xlabel("x")
    ylabel("t")
    zlabel("u")
    view(10, 70)
    return gcf()
end

# p7 - accuracy of periodic spectral differentiation
function p7()
    # Compute derivatives for various values of N:
    Nmax = 50
    allN = 6:2:Nmax
    E = zeros(4, length(allN))
    for N = 6:2:Nmax
        h = 2π / N
        x = h * (1:N)
        column = [0; @. 0.5 * (-1)^(1:N-1) * cot((1:N-1) * h / 2)]
        D = toeplitz(column, column[[1; N:-1:2]])
        v = @. abs(sin(x))^3                     # 3rd deriv in BV
        vʹ = @. 3sin(x) * cos(x) * abs(sin(x))
        j = round(Int, N / 2 - 2)
        E[1, j] = norm(D * v - vʹ, Inf)
        v = @. exp(-sin(x / 2)^(-2))               # C-infinity
        vʹ = @. 0.5v * sin(x) / sin(x / 2)^4
        E[2, j] = norm(D * v - vʹ, Inf)
        v = @. 1 / (1 + sin(x / 2)^2)                 # analytic in a strip
        vʹ = @. -sin(x / 2) * cos(x / 2) * v^2
        E[3, j] = norm(D * v - vʹ, Inf)
        v = sin.(10x)
        vʹ = 10cos.(10x)   # band-limited
        E[4, j] = norm(D * v - vʹ, Inf)
    end

    # Plot results:
    titles = [L"|\sin(x)|^3", L"\exp(-\sin^{-2}(x/2))", L"1/(1+\sin^2(x/2))", L"\sin(10x)"]
    clf()
    for iplot = 1:4
        subplot(2, 2, iplot)
        semilogy(allN, E[iplot, :], ".-", markersize=6)
        axis([0, Nmax, 1e-16, 1e3])
        grid(true)
        xticks(0:10:Nmax)
        yticks(10.0 .^ (-15:5:0))
        title(titles[iplot])
        iplot > 2 ? xlabel("N") : nothing
        iplot % 2 > 0 ? ylabel("error") : nothing
    end
    return gcf()
end

# p8 - eigenvalues of harmonic oscillator -u"+x^2 u on R
function p8()
    L = 8                             # domain is [-L L], periodic
    for N = 6:6:36
        h = 2π / N
        x = h * (1:N)
        x = @. L * (x - π) / π
        column = [-π^2 / 3h^2 - 1 / 6; @. -0.5 * (-1)^(1:N-1) / sin(h * (1:N-1) / 2)^2]
        D² = (π / L)^2 * toeplitz(column)  # 2nd-order differentiation
        eigenvalues = sort(eigvals(-D² + diagm(x .^ 2)))
        @show N
        foreach(println,eigenvalues[1:4])
        println("")
    end
end

# p9 - polynomial interpolation in equispaced and Chebyshev pts
function p9()
    N = 16
    xx = -1.01:0.005:1.01
    clf()
    for i = 1:2
        i == 1 && ((s, x) = ("equispaced points", -1 .+ 2 * (0:N) / N))
        i == 2 && ((s, x) = ("Chebyshev points", cospi.((0:N) / N)))
        subplot(2, 2, i)
        u = @. 1 / (1 + 16x^2)
        uu = @. 1 / (1 + 16xx^2)
        p = fit(x, u)              # interpolation
        pp = p.(xx)                    # evaluation of interpolant
        plot(x, u, ".", markersize=6)
        plot(xx, pp)
        axis([-1.1, 1.1, -1, 1.5])
        title(s)
        error = round(norm(uu - pp, Inf), sigdigits=5)
        text(-0.5, -0.5, "max error = $error", fontsize=8)
    end
    return gcf()
end

# p10 - polynomials and corresponding equipotential curves
function p10()
    N = 16
    clf()
    xx = -1.01:0.005:1.01
    for i = 1:2
        i == 1 && ((s, x) = ("equispaced points", -1 .+ 2 * (0:N) / N))
        i == 2 && ((s, x) = ("Chebyshev points", cospi.((0:N) / N)))
        p = fromroots(x)

        # Plot p(x) over [-1,1]:
        xx = -1:0.005:1
        pp = p.(xx)
        subplot(2, 2, 2i - 1)
        plot(x, zero(x), "k.", markersize=6)
        plot(xx, pp)
        grid(true)
        xticks(-1:0.5:1)
        title(s)

        # Plot equipotential curves:
        subplot(2, 2, 2i)
        plot(real(x), imag(x), ".", markersize=6)
        axis([-1.4, 1.4, -1.12, 1.12])
        xx = -1.4:0.02:1.4
        yy = -1.12:0.02:1.12
        zz = [complex(x, y) for x in xx, y in yy]
        pp = p.(zz)
        levels = 10.0 .^ (-4:0)
        contour(xx, yy, abs.(pp)', levels, colors="k")
        title(s)
    end
    return gcf()
end

# p11 - Chebyshev differentation of a smooth function
function p11()
    xx = -1:0.01:1
    uu = @. exp(xx) * sin(5xx)
    clf()
    for N = [10, 20]
        D, x = cheb(N)
        u = @. exp(x) * sin(5x)
        PyPlot.axes([0.15, 0.66 - 0.4 * (N == 20), 0.31, 0.28])
        plot(x, u, ".", markersize=6)
        grid(true)
        plot(xx, uu)
        title("u(x),  N=$N")
        error = D * u - @. exp(x) * (sin(5x) + 5cos(5x))
        PyPlot.axes([0.55, 0.66 - 0.4 * (N == 20), 0.31, 0.28])
        plot(x, error, ".-", markersize=6)
        grid(true)
        title("error in u'(x),  N=$N")
    end
    return gcf()
end

# p12 - accuracy of Chebyshev spectral differentiation
function p12()
    # Compute derivatives for various values of N:
    Nmax = 50
    E = zeros(4, Nmax)
    for N = 1:Nmax
        D, x = cheb(N)
        v = @. abs(x)^3
        vʹ = @. 3x * abs(x)   # 3rd deriv in BV
        E[1, N] = norm(D * v - vʹ, Inf)
        v = @. exp(-x^(-2))
        vʹ = @. 2v / x^3  # C-infinity
        E[2, N] = norm(D * v - vʹ, Inf)
        v = @. 1 / (1 + x^2)
        vʹ = @. -2x * v^2    # analytic in [-1,1]
        E[3, N] = norm(D * v - vʹ, Inf)
        v = x .^ 10
        vʹ = 10x .^ 9               # polynomial
        E[4, N] = norm(D * v - vʹ, Inf)
    end

    # Plot results:
    titles = [L"|x|^3", L"\exp(-x^2)", L"1/(1+x^2)", L"x^{10}"]
    clf()
    for iplot = 1:4
        subplot(2, 2, iplot)
        semilogy(1:Nmax, E[iplot, :], ".-", markersize=6)
        axis([0, Nmax, 1e-16, 1e3])
        grid(true)
        xticks(0:10:Nmax)
        yticks(10.0 .^ (-15:5:0))
        title(titles[iplot])
        iplot > 2 ? xlabel("N") : nothing
        iplot % 2 > 0 ? ylabel("error") : nothing
    end
    return gcf()
end

# p13 - solve linear BVP u_xx = exp(4x), u(-1)=u(1)=0
function p13()
    N = 16
    D, x = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]                   # boundary conditions
    f = @. exp(4x[2:N])
    u = D² \ f                           # Poisson eq. solved here
    u = [0; u; 0]
    clf()
    axes([0.1, 0.4, 0.8, 0.5])
    plot(x, u, ".", markersize=6)
    xx = -1:0.01:1
    uu = fit(x, u).(xx)      # interpolate grid data
    plot(xx, uu)
    grid(true)
    exact = @. (exp(4xx) - sinh(4) * xx - cosh(4)) / 16
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=4))", fontsize=12)
    return gcf()
end

# p14 - solve nonlinear BVP u_xx = exp(u), u(-1)=u(1)=0
function p14()
    N = 16
    D, x = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]
    u = zeros(N - 1)
    change = 1
    it = 0
    while change > 1e-15                   # fixed-point iteration
        unew = D² \ exp.(u)
        change = norm(unew - u, Inf)
        u = unew
        it += 1
    end
    u = [0; u; 0]
    clf()
    PyPlot.axes([0.1, 0.4, 0.8, 0.5])
    plot(x, u, ".", markersize=6)
    xx = -1:0.01:1
    uu = fit(x, u).(xx)
    plot(xx, uu), grid(true)
    title("no. steps = $it      u(0) =$(u[Int(N/2+1)])")
    return gcf()
end

# p15 - solve eigenvalue BVP u_xx = λ*u, u(-1)=u(1)=0
function p15()
    N = 36
    D, x = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]
    λ, V = eigen(D²)
    ii = sortperm(-λ)          # sort eigenvalues and -vectors
    λ = λ[ii]
    V = V[:, ii]
    clf()
    for j = 5:5:30                  # plot 6 eigenvectors
        u = [0; V[:, j]; 0]
        subplot(7, 1, j ÷ 5)
        plot(x, u, ".", markersize=6)
        grid(true)
        xx = -1:0.01:1
        uu = fit(x, u).(xx)
        plot(xx, uu)
        axis("off")
        text(-0.4, 0.1, "eig $j = $(λ[j]*4/π^2) π^2/4")
        text(0.7, 0.1, "$(round(4*N/(π*j),sigdigits=2))  ppw")
    end
    return gcf()
end

# p16 - Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary
function p16()
    # Set up grids and tensor product Laplacian and solve for u:
    N = 24
    D, x = cheb(N)
    y = x
    xx = x[2:N]
    yy = y[2:N]
    f = @. 10sin(8xx' * (yy - 1))
    D² = D^2
    D² = D²[2:N, 2:N]
    L = kron(I(N - 1), D²) + kron(D², I(N - 1))                       # Laplacian
    fig1 = figure(1)
    clf()
    spy(L)
    @elapsed u = L \ f[:]           # solve problem and watch the clock

    # Reshape long 1D results onto 2D grid (flipping orientation):
    uu = zeros(N + 1, N + 1)
    uu[N:-1:2, N:-1:2] = reshape(u, N - 1, N - 1)
    value = uu[3N ÷ 4 + 1, 3N ÷ 4 + 1]

    # Interpolate to finer grid and plot:
    xxx = yyy = -1:0.04:1
    s = Spline2D(x[end:-1:1], y[end:-1:1], uu, kx=1, ky=1)
    uuu = s.(xxx, yyy')
    fig2 = figure(2)
    clf()
    surf(xxx, yyy, uuu', rstride=1, cstride=1)
    xlabel("x")
    ylabel("y")
    zlabel("u")
    view(-37.5, 30)
    text3D(0.4, -0.3, -0.3, "\$u(2^{-1/2},2^{-1/2})\$ = $(round(value,sigdigits=11))", fontsize=9)
    return [fig1,fig2]
end

# p17 - Helmholtz eq. u_xx + u_yy + (k^2)u = f
function p17()
    # Set up spectral grid and tensor product Helmholtz operator:
    N = 24
    D, x = cheb(N)
    y = x
    xx = x[2:N]
    yy = y[2:N]
    f = @. exp(-10 * ((yy - 1)^2 + (xx' - 0.5)^2))
    D² = D^2
    D² = D²[2:N, 2:N]
    k = 9
    L = kron(I(N - 1), D²) + kron(D², I(N - 1)) + k^2 * I((N - 1)^2)

    # Solve for u, reshape to 2D grid, and plot:
    u = L \ f[:]
    uu = zeros(N + 1, N + 1)
    uu[N:-1:2, N:-1:2] = reshape(u, N - 1, N - 1)
    xxx = yyy = -1:0.0333:1
    s = Spline2D(x[end:-1:1], y[end:-1:1], uu)
    uuu = evalgrid(s, xxx, yyy)
    figure(1)
    clf()
    surf(xxx, yyy, uuu, rstride=1, cstride=1)
    xlabel("x")
    ylabel("y")
    zlabel("u")
    view(-37.5, 30)
    value = round(uu[Int(N / 2 + 1), Int(N / 2 + 1)], sigdigits=10)
    text3D(0.2, 1, 0.022, "u(0,0) = $value")
    figure(2)
    clf()
    contour(xxx, yyy, uuu, 10)
    axis("square")
    return gcf()
end

# p18 - Chebyshev differentiation via FFT (compare p11.jl)
function p18()
    xx = -1:0.01:1
    ff = @. exp(xx) * sin(5xx)
    clf()
    for N = [10 20]
        _, x = cheb(N)
        f = @. exp(x) * sin(5x)
        PyPlot.axes([0.15, 0.66 - 0.4 * (N == 20), 0.31, 0.28])
        plot(x, f, "k.", markersize=6)
        grid(true)
        plot(xx, ff)
        title("f(x), N=$N")
        error = chebfft(f) - @. exp(x) * (sin(5x) + 5cos(5x))
        PyPlot.axes([0.55, 0.66 - 0.4 * (N == 20), 0.31, 0.28])
        plot(x, error, ".-", markersize=10)
        grid(true)
        title("error in f'(x),  N=$N")
    end
    return gcf()
end

# p19 - 2nd-order wave eq. on Chebyshev grid (compare p6.jl)
function p19()
    # Time-stepping by leap frog formula:
    N = 80
    _, x = cheb(N)
    Δt = 8 / N^2
    v = @. exp(-200x^2)
    vold = @. exp(-200 * (x - Δt)^2)
    tmax = 4
    tplot = 0.075
    plotgap = round(Int, tplot / Δt)
    Δt = tplot / plotgap
    nplots = round(Int, tmax / tplot)
    plotdata = [v zeros(N + 1, nplots)]
    tdata = 0
    for i = 1:nplots
        for n = 1:plotgap
            w = chebfft(chebfft(v))
            w[1] = 0
            w[N+1] = 0
            vnew = 2v - vold + Δt^2 * w
            vold = v
            v = vnew
        end
        plotdata[:, i+1] = v
        tdata = [tdata; Δt * i * plotgap]
    end

    # Plot results:
    clf()
    mesh(x, tdata, plotdata', ccount=0, rcount=N + 1)
    axis([-1, 1, 0, tmax])
    zlim(-2, 2)
    view(10, 70)
    grid(false)
    xlabel("x")
    ylabel("t")
    zlabel("u")
    return gcf()
end

# p20 - 2nd-order wave eq. in 2D via FFT (compare p19.m)
function p20()
    # Grid and initial data:
    N = 24
    _, x = cheb(N)
    y = x
    Δt = 6 / N^2
    xx = yy = x[end:-1:1]
    plotgap = round(Int, (1 / 3) / Δt)
    Δt = (1 / 3) / plotgap
    vv = @. exp(-40 * ((x' - 0.4)^2 + y^2))
    vvold = vv
    clf()

    # Time-stepping by leap frog formula:
    for n = 0:3*plotgap
        t = n * Δt
        if rem(n + 0.5, plotgap) < 1     # plots at multiples of t=1/3
            i = n ÷ plotgap + 1
            subplot(2, 2, i, projection="3d")
            xxx = yyy = -1:1/16:1
            s = Spline2D(xx, yy, reverse(vv, dims=:))
            vvv = evalgrid(s, xxx, yyy)
            surf(xxx, yyy, vvv)
            xlim(-1, 1)
            ylim(-1, 1)
            zlim(-0.15, 1)
            view(-37.5, 30)
            title("t = $(round(t,sigdigits=5))")
        end
        uxx = zeros(N + 1, N + 1)
        uyy = zeros(N + 1, N + 1)
        ii = 2:N
        for i = 2:N                # 2nd derivs wrt x in each row
            v = vv[i, :]
            V = [v; reverse(v[ii])]
            U = real(fft(V))
            W1 = real(ifft(1im * [0:N-1; 0; 1-N:-1] .* U)) # diff wrt θ
            W2 = real(ifft(-[0:N; 1-N:-1] .^ 2 .* U))     # diff^2 wrt θ
            uxx[i, ii] = W2[ii] ./ (1 .- x[ii] .^ 2) - x[ii] .* W1[ii] ./ (1 .- x[ii] .^ 2) .^ (3 / 2)
        end
        for j = 2:N                # 2nd derivs wrt y in each column
            v = vv[:, j]
            V = [v; reverse(v[ii])]
            U = real(fft(V))
            W1 = real(ifft(1im * [0:N-1; 0; 1-N:-1] .* U))# diff wrt θ
            W2 = real(ifft(-[0:N; 1-N:-1] .^ 2 .* U))    # diff^2 wrt θ
            uyy[ii, j] = W2[ii] ./ (1 .- y[ii] .^ 2) - y[ii] .* W1[ii] ./ (1 .- y[ii] .^ 2) .^ (3 / 2)
        end
        vvnew = 2vv - vvold + Δt^2 * (uxx + uyy)
        vvold = vv
        vv = vvnew
    end
    return gcf()
end

# p21 - eigenvalues of Mathieu operator -u_xx + 2qcos(2x)u
function p21()
    N = 42
    h = 2π / N
    x = h * (1:N)
    D² = toeplitz([-π^2 / (3h^2) - 1 / 6; @. -0.5 * (-1)^(1:N-1) / sin(h * (1:N-1) / 2)^2])
    qq = 0:0.2:15
    data = zeros(0, 11)
    for q = qq
        e = sort(eigvals(-D² + 2q * diagm(cos.(2x))))
        data = [data; e[1:11]']
    end
    clf()
    subplot(1, 2, 1)
    plot(qq, data[:, 1:2:end], "b-")
    plot(qq, data[:, 2:2:end], "b--")
    xlabel("q")
    ylabel("λ")
    axis([0, 15, -24, 32])
    yticks(-24:4:32)
    return gcf()
end

# p22 - 5th eigenvector of Airy equation u_xx = λ*x*u
function p22()
    clf()
    for N = 12:12:48
        D, x = cheb(N)
        D² = D^2
        D² = D²[2:N, 2:N]
        λ, V = eigen(D², diagm(x[2:N]))      # generalized ev problem
        ii = findall(λ .> 0)
        V = V[:, ii]
        λ = λ[ii]
        ii = sortperm(λ)[5]
        λ = λ[ii]
        v = [0; V[:, ii]; 0]
        v = v / v[Int(N / 2 + 1)] * airyai(0)
        xx = -1:0.01:1
        vv = fit(x, v).(xx)
        subplot(2, 2, N ÷ 12)
        plot(xx, vv)
        grid(true)
        title("N = $N     eig = $(round(λ,sigdigits=13))")
    end
    return gcf()
end

# p23 - eigenvalues of perturbed Laplacian on [-1,1]x[-1,1]
function p23()
    # Set up tensor product Laplacian and compute 4 eigenmodes:
    N = 16
    D, x = cheb(N)
    y = x
    xx = x[2:N]
    yy = y[2:N]
    D² = D^2
    D² = D²[2:N, 2:N]
    L = -kron(I(N - 1), D²) - kron(D², I(N - 1))                #  Laplacian
    f = @. exp(20 * (yy - xx' - 1))      # perturbation
    L = L + diagm(f[:])
    D, V = eigen(L)
    ii = sortperm(D)[1:4]
    D = D[ii]
    V = V[:, ii]

    # Reshape them to 2D grid, interpolate to finer grid, and plot:
    xx = yy = x[end:-1:1]
    xxx = yyy = -1:0.02:1
    uu = zeros(N + 1, N + 1)
    ay, ax = (repeat([0.56 0.04], outer=(2, 1)), repeat([0.1, 0.5], outer=(1, 2)))
    clf()
    for i = 1:4
        uu[2:N, 2:N] = reshape(V[:, i], N - 1, N - 1)
        uu = uu / norm(uu[:], Inf)
        s = Spline2D(xx, yy, reverse(uu, dims=:))
        uuu = evalgrid(s, xxx, yyy)
        PyPlot.axes([ax[i], ay[i], 0.38, 0.38])
        contour(xxx, yyy, uuu, levels=-0.9:0.2:0.9)
        axis("square")
        title("eig = $(round(D[i]/(π^2/4),sigdigits=13)) π^2/4")
    end
    return gcf()
end

# p23a - eigenvalues of UNperturbed Laplacian on [-1,1]x[-1,1]
function p23a()
    # Set up tensor product Laplacian and compute 4 eigenmodes:
    N = 16
    D, x = cheb(N)
    y = x
    xx = x[2:N]
    yy = y[2:N]
    D² = D^2
    D² = D²[2:N, 2:N]
    L = -kron(I(N - 1), D²) - kron(D², I(N - 1))                #  Laplacian
    #f = @. exp(20*(yy-xx'-1));      # perturbation
    #L = L + diagm(f[:]);
    D, V = eigen(L)
    ii = sortperm(D)[1:4]
    D = D[ii]
    V = V[:, ii]

    # Reshape them to 2D grid, interpolate to finer grid, and plot:
    xx = yy = x[end:-1:1]
    xxx = yyy = -1:0.02:1
    uu = zeros(N + 1, N + 1)
    ay, ax = (repeat([0.56 0.04], outer=(2, 1)), repeat([0.1, 0.5], outer=(1, 2)))
    clf()
    for i = 1:4
        uu[2:N, 2:N] = reshape(V[:, i], N - 1, N - 1)
        uu = uu / norm(uu[:], Inf)
        s = Spline2D(xx, yy, reverse(uu, dims=:))
        uuu = evalgrid(s, xxx, yyy)
        PyPlot.axes([ax[i], ay[i], 0.38, 0.38])
        contour(xxx, yyy, uuu, levels=-0.9:0.2:0.9)
        axis("square")
        title("eig = $(round(D[i]/(π^2/4),sigdigits=13)) π^2/4")
    end
    return gcf()
end

# p24 - pseudospectra of Davies's complex harmonic oscillator
function p24()
    # Eigenvalues:
    N = 70
    D, x = cheb(N)
    x = x[2:N]
    L = 6
    x = L * x
    D = D / L                   # rescale to [-L,L]
    A = -D^2
    A = A[2:N, 2:N] + (1 + 3im) * diagm(x .^ 2)
    λ = eigvals(A)
    clf()
    plot(real(λ), imag(λ), "k.", markersize=6)
    axis([0, 50, 0, 40])

    # Pseudospectra:
    x = 0:1:50
    y = 0:1:40
    zz = x' .+ 1im * y
    minsvd(z) = minimum(svdvals(z * I - A))
    sigmin = [minsvd(x[i] + 1im * y[j]) for i = eachindex(x), j = eachindex(y)]
    contour(x, y, sigmin', levels=10.0 .^ (-4:0.5:-0.5))
    return gcf()
end

# p24 - pseudospectra of Davies's complex harmonic oscillator
function p24fine()
    # Eigenvalues:
    N = 70
    D, x = cheb(N)
    x = x[2:N]
    L = 6
    x = L * x
    D = D / L                   # rescale to [-L,L]
    A = -D^2
    A = A[2:N, 2:N] + (1 + 3im) * diagm(x .^ 2)
    λ = eigvals(A)
    clf()
    plot(real(λ), imag(λ), "k.", markersize=6)
    axis([0, 50, 0, 40])

    # Pseudospectra:
    x = 0:0.5:50
    y = 0:0.5:40
    minsvd(z) = minimum(svdvals(z * I - A))
    sigmin = [minsvd(x[i] + 1im * y[j]) for i = eachindex(x), j = eachindex(y)]
    contour(x, y, sigmin', levels=10.0 .^ (-4:0.5:-0.5))
    return gcf()
end

# p25 - stability regions for ODE formulas
function p25()
    # Adams-Bashforth:
    clf()
    subplot(221)
    zplot(z) = plot(real(z), imag(z))
    plot([-8, 8], [0, 0], "k-")
    plot([0, 0], [-8, 8], "k-")
    z = exp.(1im * π * (0:200) / 100)
    r = z .- 1
    s = 1
    zplot(r ./ s)                                  # order 1
    s = @. (3 - 1 / z) / 2
    zplot(r ./ s)                         # order 2
    s = @. (23 - 16 / z + 5 / z^2) / 12
    zplot(r ./ s)             # order 3
    axis("square")
    axis([-2.5, 0.5, -1.5, 1.5])
    grid(true)
    title("Adams—Bashforth")

    # Adams-Moulton:
    subplot(222)
    plot([-8, 8], [0, 0], "k-")
    plot([0, 0], [-8, 8], "k-")
    s = @. (5z + 8 - 1 / z) / 12
    zplot(r ./ s)                    # order 3
    s = @. (9z + 19 - 5 / z + 1 / z^2) / 24
    zplot(r ./ s)           # order 4
    s = @. (251z + 646 - 264 / z + 106 / z^2 - 19 / z^3) / 720
    zplot(r ./ s)   # 5
    d = @. 1 - 1 / z
    s = @. 1 - d / 2 - d^2 / 12 - d^3 / 24 - 19d^4 / 720 - 3d^5 / 160
    zplot(d ./ s) # 6
    axis("square")
    axis([-7, 1, -4, 4])
    grid(true)
    title("Adams—Moulton")

    # Backward differentiation:
    subplot(223)
    plot([-40, 40], [0, 0], "k-")
    plot([0, 0], [-40, 40], "k-")
    r = zeros(size(d))
    for i = 1:6  # orders 1-6
        r += @. (d^i) / i
        zplot(r)
    end
    axis("square")
    axis([-15, 35, -25, 25])
    grid(true)
    title("backward differentiation")

    # Runge-Kutta:
    subplot(224)
    plot([-8, 8], [0, 0], "k-")
    plot([0, 0], [-8, 8], "k-")
    w = complex(zeros(4))
    W = transpose(w)
    for i = 2:length(z)
        # orders 1-4
        w[1] -= (1 + w[1] - z[i])
        w[2] -= (1 + w[2] + 0.5w[2]^2 - z[i]^2) / (1 + w[2])
        w[3] -= (1 + w[3] + 0.5w[3]^2 + w[3]^3 / 6 - z[i]^3) / (1 + w[3] + w[3]^2 / 2)
        w[4] -= (1 + w[4] + 0.5w[4]^2 + w[4]^3 / 6 + w[4] .^ 4 / 24 - z[i]^4) / (1 + w[4] + w[4]^2 / 2 + w[4]^3 / 6)
        W = [W; transpose(w)]
    end
    zplot(W)
    axis("square")
    axis([-5, 2, -3.5, 3.5])
    grid(true)
    title("Runge—Kutta")
    return gcf()
end

# p26 - eigenvalues of 2nd-order Chebyshev diff. matrix
function p26()
    N = 60
    D, x = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]
    λ, V = eigen(D²)
    ii = sortperm(-λ)
    e = λ[ii]
    V = V[:, ii]

    # Plot eigenvalues:
    clf()
    PyPlot.axes([0.1, 0.62, 0.8, 0.3])
    loglog(-e, ".", markersize=4)
    ylabel("eigenvalue")
    title("N = $N       max |λ| = $(round(maximum(-e)/N^4,sigdigits=5)) \$N^4\$")
    semilogy(2N / π * [1, 1], [1, 1e6], "--r")
    text(2.1N / π, 24, "2π / N", fontsize=12)

    # Plot eigenmodes N/4 (physical) and N (nonphysical):
    vN4 = [0; V[:, Int(N / 4 - 1)]; 0]
    xx = -1:0.01:1
    vv = fit(x, vN4).(xx)
    PyPlot.axes([0.1, 0.36, 0.8, 0.15])
    plot(xx, vv)
    plot(x, vN4, "k.", markersize=4)
    title("eigenmode N/4")
    vN = V[:, N-1]
    PyPlot.axes([0.1, 0.1, 0.8, 0.15])
    semilogy(x[2:N], abs.(vN))
    axis([-1, 1, 5e-6, 1])
    plot(x[2:N], abs.(vN), "k.", markersize=4)
    title("absolute value of eigenmode N    (log scale)")
    return gcf()
end

# p27 - Solve KdV eq. u_t + uu_x + u_xxx = 0 on [-π,π] by
function p27()
    # Set up grid and two-soliton initial data:
    N = 256
    Δt = 0.4 / N^2
    x = (2π / N) * (-N/2:N/2-1)
    A = 25
    B = 16
    clf()
    u = @. 3A^2 * sech(0.5 * (A * (x + 2)))^2 + 3B^2 * sech(0.5 * (B * (x + 1)))^2
    v = fft(u)
    k = [0:N/2-1; 0; -N/2+1:-1]
    ik3 = 1im * k .^ 3

    # Solve PDE and plot results:
    tmax = 0.006
    nplt = floor(Int, (tmax / 25) / Δt)
    nmax = round(Int, tmax / Δt)
    udata = u
    tdata = [0.0]
    for n = 1:nmax
        t = n * Δt
        g = -0.5im * Δt * k
        E = exp.(Δt * ik3 / 2)
        E2 = E .^ 2
        a = g .* fft(real(ifft(v)) .^ 2)
        b = g .* fft(real(ifft(E .* (v + a / 2))) .^ 2)     # 4th-order
        c = g .* fft(real(ifft(E .* v + b / 2)) .^ 2)     # Runge-Kutta
        d = g .* fft(real(ifft(E2 .* v + E .* c)) .^ 2)
        v = E2 .* v + (E2 .* a + 2E .* (b + c) + d) / 6
        if mod(n, nplt) == 0
            u = real(ifft(v))
            udata = [udata u]
            tdata = [tdata; t]
        end
    end
    mesh(x, tdata, udata', ccount=0, rcount=N)
    view(-20, 25)
    xlabel("x")
    ylabel("y")
    grid(true)
    xlim(-π, π)
    ylim(0, tmax)
    zlim(0, 12000)
    |
    zticks([0, 2000])
    return gcf()
end

# p28 - eigenmodes of Laplacian on the disk (compare p22.jl)
function p28()
    # r coordinate, ranging from -1 to 1 (N must be odd):
    N = 25
    N2 = Int((N - 1) / 2)
    D, r = cheb(N)
    D² = D^2
    D1 = D²[2:N2+1, 2:N2+1]
    D² = D²[2:N2+1, N:-1:N2+2]
    E1 = D[2:N2+1, 2:N2+1]
    E2 = D[2:N2+1, N:-1:N2+2]

    # t = θ coordinate, ranging from 0 to 2*π (M must be even):
    M = 20
    Δt = 2π / M
    t = Δt * (1:M)
    M2 = Int(M / 2)
    D²t = toeplitz([-π^2 / (3Δt^2) - 1 / 6; @. 0.5 * (-1)^(2:M) / sin(Δt * (1:M-1) / 2)^2])

    # Laplacian in polar coordinates:
    R = diagm(1 ./ r[2:N2+1])
    Z = zeros(M2, M2)
    L = kron(D1 + R * E1, I(M)) + kron(D² + R * E2, [Z I(M2); I(M2) Z]) + kron(R^2, D²t)

    # Compute four eigenmodes:
    index = [1, 3, 6, 10]
    λ, V = eigen(-L)
    ii = sortperm(abs.(λ))[index]
    λ = λ[ii]
    V = V[:, ii]
    λ = sqrt.(real(λ / λ[1]))

    # Plot eigenmodes with nodal lines underneath:
    (rr, tt) = (r[1:N2+1], [0; t])
    (xx, yy) = @. (cos(tt) * rr', sin(tt) * rr')
    z = exp.(1im * π * (-100:100) / 100)
    for i = 1:4
        figure(i)
        clf()
        u = reshape(real(V[:, i]), M, N2)
        u = [zeros(M + 1) u[[M; 1:M], :]]
        u = u / norm(u[:], Inf)
        #plot3D(real(z),imag(z),zeros(size(z)));
        xlim(-1.05, 1.05)
        ylim(-1.05, 1.05)
        zlim(-1.05, 1.05)
        axis("off")
        surf(xx, yy, u)
        contour3D(xx, yy, u .- 1, levels=[-1])
        plot3D(real(z), imag(z), -abs.(z))
        title("Mode $(index[i]):  λ = $(round(λ[i],sigdigits=11))", fontsize=9)
    end
    return gcf()
end

# p28b - eigenmodes of Laplacian on the disk
function p28b()
    # r coordinate, ranging from -1 to 1 (N must be odd)
    N = 25
    N2 = Int((N - 1) / 2)
    D, r = cheb(N)
    D² = D^2
    D1 = D²[2:N2+1, 2:N2+1]
    D² = D²[2:N2+1, N:-1:N2+2]
    E1 = D[2:N2+1, 2:N2+1]
    E2 = D[2:N2+1, N:-1:N2+2]

    # t = θ coordinate, ranging from 0 to 2*π (M must be even):
    M = 20
    Δt = 2π / M
    t = Δt * (1:M)
    M2 = Int(M / 2)
    D²t = toeplitz([-π^2 / (3Δt^2) - 1 / 6; @. 0.5 * (-1)^(2:M) / sin(Δt * (1:M-1) / 2)^2])

    # Laplacian in polar coordinates:
    R = diagm(1 ./ r[2:N2+1])
    Z = zeros(M2, M2)
    L = kron(D1 + R * E1, I(M)) + kron(D² + R * E2, [Z I(M2); I(M2) Z]) + kron(R^2, D²t)

    # Compute 25 eigenmodes:
    index = 1:25
    λ, V = eigen(-L)
    ii = sortperm(abs.(λ))[index]
    λ = λ[ii]
    V = V[:, ii]
    λ = sqrt.(real(λ / λ[1]))

    # Plot nodal lines:
    (rr, tt) = (r[1:N2+1], [0; t])
    (xx, yy) = @. (cos(tt) * rr', sin(tt) * rr')
    z = exp.(1im * π * (-100:100) / 100)
    clf()
    for i = 1:25
        subplot(5, 5, i)
        u = reshape(real(V[:, i]), M, N2)
        u = [zeros(M + 1) u[[M; 1:M], :]]
        u = u / norm(u[:], Inf)
        plot(real(z), imag(z))
        xlim(-1.07, 1.07)
        ylim(-1.07, 1.07)
        axis("off")
        axis("equal")
        contour(xx, yy, u, levels=[0])
        title("$(round(λ[i],sigdigits=5))", fontsize=8)
    end
    return gcf()
end

# p29 - solve Poisson equation on the unit disk
function p29()
    # Laplacian in polar coordinates:
    N = 25
    N2 = Int((N - 1) / 2)
    D, r = cheb(N)
    D² = D^2
    D1 = D²[2:N2+1, 2:N2+1]
    D² = D²[2:N2+1, N:-1:N2+2]
    E1 = D[2:N2+1, 2:N2+1]
    E2 = D[2:N2+1, N:-1:N2+2]
    M = 20
    Δt = 2π / M
    t = Δt * (1:M)
    M2 = Int(M / 2)
    D²t = toeplitz([-π^2 / (3Δt^2) - 1 / 6; @. 0.5 * (-1)^(2:M) / sin(Δt * (1:M-1) / 2)^2])
    R = diagm(1 ./ r[2:N2+1])
    Z = zeros(M2, M2)
    L = kron(D1 + R * E1, I(M)) + kron(D² + R * E2, [Z I(M2); I(M2) Z]) + kron(R^2, D²t)

    # Right-hand side and solution for u:
    (rr, tt) = (r[2:N2+1]', t)
    f = @. -rr^2 * sin(tt / 2)^4 + sin(6tt) * cos(tt / 2)^2
    u = L \ f[:]

    # Reshape results onto 2D grid and plot them:
    u = reshape(u, M, N2)
    u = [zeros(M + 1) u[[M; 1:M], :]]
    (rr, tt) = (r[1:N2+1], t[[M; 1:M]])
    (xx, yy) = @. (cos(tt) * rr', sin(tt) * rr')
    clf()
    surf(xx, yy, u)
    view(20, 40)
    xlim(-1, 1)
    ylim(-1, 1)
    zlim(-0.01, 0.05)
    xlabel("x")
    ylabel("y")
    zlabel("z")
    return gcf()
end

function _p30(method)
    Nmax = 50
    # Computation: various values of N, four functions:
    clf()
    E = zeros(4, Nmax)
    for N = 1:Nmax
        x, w = method(N)
        f = @. abs(x)^3
        E[1, N] = abs(dot(w, f) - 0.5)
        f = @. exp(-x^(-2))
        E[2, N] = abs(dot(w, f) - 2 * (exp(-1) + sqrt(π) * (erf(1) - 1)))
        f = @. 1 / (1 + x^2)
        E[3, N] = abs(dot(w, f) - π / 2)
        f = x .^ 10
        E[4, N] = abs(dot(w, f) - 2 / 11)
    end

    # Plot results:
    labels = [L"|x|^3", L"\exp(-x^2)", L"1/(1+x^2)", L"x^{10}"]
    for iplot = 1:4
        subplot(2, 2, iplot)
        semilogy(E[iplot, :] .+ 1e-100, ".-", markersize=10)
        axis([0, Nmax, 1e-18, 1e3])
        grid(true)
        xticks(0:10:Nmax)
        yticks((10.0) .^ (-15:5:0))
        ylabel("error")
        text(32, 0.004, labels[iplot])
    end
    return gcf()
end

# p30 - spectral integration, ODE style (compare p12.jl)
function p30()
    _p30(
        function (N)
            i = 1:N
            D, x = cheb(N)
            x = x[i]
            Di = inv(D[i, i])
            w = Di[1, :]
            return x, w
        end
    )
end

# p30b - spectral integration, Clenshaw-Curtis style (compare p30.jl)
p30b() = _p30(clencurt)

# p30c - spectral integration, Gauss style (compare p30.jl)
p30c() = _p30(gauss)

# p31 - gamma function via complex integral, trapezoid rule
function p31()
    N = 70
    θ = @. -π + (2π / N) * (0.5:N-0.5)
    c = -11                     # center of circle of integration
    r = 16                      # radius of circle of integration
    x = -3.5:0.1:4
    y = -2.5:0.1:2.5
    zz = x' .+ 1im * y
    gaminv = 0zz
    for i = 1:N
        t = c + r * cis(θ[i])
        gaminv += exp(t) * t .^ (-zz) * (t - c)
    end
    gaminv = gaminv / N
    Γ = 1 ./ gaminv
    clf()
    surf(x, y, abs.(Γ))
    xlim(-3.5, 4)
    ylim(-2.5, 2.5)
    zlim(0, 6)
    xlabel("Re(z)")
    ylabel("Im(z)")
    text3D(4, -1.4, 5.5, "\$|\\Gamma(z)|\$", fontsize=20)
    return gcf()
end

# p32 - solve u_xx = exp(4x), u(-1)=0, u(1)=1 (compare p13.jl)
function p32()
    N = 16
    D, x = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]                   # boundary conditions
    f = @. exp(4x[2:N])
    u = D² \ f                           # Poisson eq. solved here
    u = [0; u; 0] + (x .+ 1) / 2
    clf()
    plot(x, u, ".", markersize=10)
    xx = -1:0.01:1
    uu = fit(x, u).(xx)
    plot(xx, uu)
    grid(true)
    exact = @. (exp(4xx) - sinh(4) * xx - cosh(4)) / 16 + (xx + 1) / 2
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=4))", fontsize=12)
    return gcf()
end

# p33 - solve linear BVP u_xx = exp(4x), u'(-1)=u(1)=0
function p33()
    N = 16
    D, x = cheb(N)
    D² = D^2
    D²[N+1, :] = D[N+1, :]            # Neumann condition at x = -1
    D² = D²[2:N+1, 2:N+1]
    f = @. exp(4x[2:N])
    u = D² \ [f; 0]
    u = [0; u]
    clf()
    plot(x, u, ".", markersize=10)
    axis([-1, 1, -4, 0])
    xx = -1:0.01:1
    uu = fit(x, u).(xx)
    plot(xx, uu)
    grid(true)
    exact = @. (exp(4xx) - 4exp(-4) * (xx - 1) - exp(4)) / 16
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=5))", fontsize=12)
    return gcf()
end

# p34 - Allen-Cahn eq. u_t = eps*u_xx+u-u^3, u(-1)=-1, u(1)=1
function p34()
    # Differentiation matrix and initial data:
    N = 20
    D, x = cheb(N)
    D² = D^2     # use full-size matrix
    D²[[1, N + 1], :] .= 0                     # for convenience
    eps = 0.01
    Δt = min(0.01, 50 / (N^4 * eps))
    t = 0.0
    v = @. 0.53x + 0.47sinpi(-1.5x)

    # Solve PDE by Euler formula and plot results:
    tmax = 100
    tplot = 2
    nplots = round(Int, tmax / tplot)
    plotgap = round(Int, tplot / Δt)
    Δt = tplot / plotgap
    xx = -1:0.025:1
    vv = fit(x, v).(xx)
    plotdata = [vv zeros(length(xx), nplots)]
    tdata = t
    for i = 1:nplots
        for n = 1:plotgap
            t = t + Δt
            v = v + Δt * (eps * D² * (v - x) + v - v .^ 3)    # Euler
        end
        vv = fit(x, v).(xx)
        plotdata[:, i+1] = vv
        tdata = [tdata; t]
    end
    clf()
    surf(xx, tdata, plotdata')
    grid(true)
    xlim(-1, 1)
    ylim(0, tmax)
    zlim(-1, 1)
    view(-60, 55)
    xlabel("x")
    ylabel("t")
    zlabel("u")
    return gcf()
end

# p35 - Allen-Cahn eq. as in p34.m, but with boundary condition
function p35()
    # Differentiation matrix and initial data:
    N = 20
    D, x = cheb(N)
    D² = D^2     # use full-size matrix
    eps = 0.01
    Δt = min(0.01, 50 / (N^4 * eps))
    t = 0.0
    v = @. 0.53x + 0.47sinpi(-1.5x)

    # Solve PDE by Euler formula and plot results:
    tmax = 100
    tplot = 2
    nplots = round(Int, tmax / tplot)
    plotgap = round(Int, tplot / Δt)
    Δt = tplot / plotgap
    xx = -1:0.025:1
    vv = fit(x, v).(xx)
    plotdata = [vv zeros(length(xx), nplots)]
    tdata = t
    for i = 1:nplots
        for n = 1:plotgap
            t = t + Δt
            v = v + Δt * (eps * D² * (v - x) + v - v .^ 3)    # Euler
            v[1] = 1 + sin(t / 5)^2
            v[end] = -1               # BC
        end
        vv = fit(x, v).(xx)
        plotdata[:, i+1] = vv
        tdata = [tdata; t]
    end
    clf()
    surf(xx, tdata, plotdata')
    grid(true)
    xlim(-1, 1)
    ylim(0, tmax)
    zlim(-1, 2)
    view(-60, 55)
    xlabel("x")
    ylabel("t")
    zlabel("u")
    return gcf()
end

# p36 - Laplace eq. on [-1,1]x[-1,1] with nonzero BCs
function p36()
    # Set up grid and 2D Laplacian, boundary points included:
    N = 24
    D, x = cheb(N)
    y = x
    xx = repeat(x', outer=(N + 1, 1))
    yy = repeat(y, outer=(1, N + 1))
    D² = D^2
    L = kron(I(N + 1), D²) + kron(D², I(N + 1))

    # Impose boundary conditions by replacing appropriate rows of L:
    b = @. (abs(xx[:]) == 1) | (abs(yy[:]) == 1)            # boundary pts
    L[b, :] .= 0
    L[b, b] = I(4N)
    rhs = zeros((N + 1)^2)
    rhs[b] = @. (yy[b] == 1) * (xx[b] < 0) * sinpi(xx[b])^4 + 0.2 * (xx[b] == 1) * sinpi(3yy[b])

    # Solve Laplace equation, reshape to 2D, and plot:
    u = L \ rhs
    uu = reshape(u, N + 1, N + 1)
    xxx = yyy = -1:0.04:1
    s = Spline2D(x[end:-1:1], y[end:-1:1], reverse(uu, dims=:), kx=1, ky=1)
    clf()
    surf(xxx, yyy, s.(xxx, yyy'))
    xlim(-1, 1)
    ylim(-1, 1)
    zlim(-0.2, 1)
    view(-20, 45)
    umid = uu[Int(N / 2)+1, Int(N / 2)+1]
    text3D(0, 0.8, 0.4, "u(0,0) = $(round(umid,sigdigits=9))")
    return gcf()
end

# p37 - 2D "wave tank" with Neumann BCs for |y|=1
function p37()
    # x variable in [-A,A], Fourier:
    A = 3
    Nx = 50
    dx = 2A / Nx
    x = @. -A + dx * (1:Nx)
    D²x = (π / A)^2 * toeplitz([-1 / (3 * (dx / A)^2) - 1 / 6
        @. 0.5 * (-1) .^ (2:Nx) / sinpi((dx / A) * (1:Nx-1) / 2)^2])

    # y variable in [-1,1], Chebyshev:
    Ny = 15
    Dy, y = cheb(Ny)
    D²y = Dy^2
    BC = -Dy[[1, Ny + 1], [1, Ny + 1]] \ Dy[[1, Ny + 1], 2:Ny]

    # Grid and initial data:
    vv = @. exp(-8 * ((x' + 1.5)^2 + y^2))
    Δt = 5 / (Nx + Ny^2)
    vvold = @. exp(-8 * ((x' + Δt + 1.5) .^ 2 + y .^ 2))

    # Time-stepping by leap frog formula:
    plotgap = round(Int, 2 / Δt)
    Δt = 2 / plotgap
    for n = 0:2*plotgap
        t = n * Δt
        if rem(n + 0.5, plotgap) < 1
            figure(n / plotgap + 1)
            clf()
            surf(x, y, vv)
            view(-10, 60)
            xlim(-A, A)
            ylim(-1, 1)
            zlim(-0.15, 1)
            text3D(-2.5, 1, 0.5, "t = $(round(t))", fontsize=18)
            zticks([])
        end
        vvnew = 2vv - vvold + Δt^2 * (vv * D²x + D²y * vv)
        vvold = vv
        vv = vvnew
        vv[[1, Ny + 1], :] = BC * vv[2:Ny, :]       # Neumann BCs for |y|=1
    end
    return gcf()
end

# p38 - solve u_xxxx = exp(x), u(-1)=u(1)=u'(-1)=u'(1)=0
function p38()
    # Construct discrete biharmonic operator:
    N = 15
    D, x = cheb(N)
    S = diagm([0; 1 ./ (1 .- x[2:N] .^ 2); 0])
    D4 = (diagm(1 .- x .^ 2) * D^4 - 8diagm(x) * D^3 - 12D^2) * S
    D4 = D4[2:N, 2:N]

    # Solve boundary-value problem and plot result:
    f = @. exp(x[2:N])
    u = D4 \ f
    u = [0; u; 0]
    clf()
    axes([0.1, 0.4, 0.8, 0.5])
    plot(x, u, ".", markersize=10)
    axis([-1, 1, -0.01, 0.06])
    grid(true)
    xx = (-1:0.01:1)
    uu = (1 .- xx .^ 2) .* fit(x, S * u).(xx)
    plot(xx, uu)

    # Determine exact solution and print maximum error:
    A = [1 -1 1 -1; 0 1 -2 3; 1 1 1 1; 0 1 2 3]
    V = xx .^ (0:3)'
    c = A \ exp.([-1, -1, 1, 1])
    exact = exp.(xx) - V * c
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=5))", fontsize=12)
    return gcf()
end

# p39 - eigenmodes of biharmonic on a square with clamped BCs
function p39()
    # Construct spectral approximation to biharmonic operator:
    N = 17
    D, x = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]
    S = diagm([0; 1 ./ (1 .- x[2:N] .^ 2); 0])
    D4 = (diagm(1 .- x .^ 2) * D^4 - 8diagm(x) * D^3 - 12D^2) * S
    D4 = D4[2:N, 2:N]
    L = kron(I(N - 1), D4) + kron(D4, I(N - 1)) + 2kron(D², I(N - 1)) * kron(I(N - 1), D²)

    # Find and plot 25 eigenmodes:
    λ, V = eigen(-L)
    λ = -real(λ)
    ii = sortperm(λ)[1:25]
    λ = λ[ii]
    V = real(V[:, ii])
    λ = sqrt.(λ / λ[1])
    y = x
    xxx = yyy = -1:0.01:1
    sq = [1 + 1im, -1 + 1im, -1 - 1im, 1 - 1im, 1 + 1im]
    clf()
    for i = 1:25
        uu = zeros(N + 1, N + 1)
        uu[2:N, 2:N] = reshape(V[:, i], N - 1, N - 1)
        subplot(5, 5, i)
        plot(real(sq), imag(sq))
        s = Spline2D(x[end:-1:1], y[end:-1:1], reverse(uu, dims=:), kx=1, ky=1)
        contour(xxx, yyy, s.(xxx, yyy'), levels=[0], color="k")
        axis("square")
        axis(1.25 * [-1, 1, -1, 1])
        axis("off")
        text(-0.3, 1.15, "$(round(λ[i],sigdigits=5))", fontsize=7)
    end
    return gcf()
end

# p40 - eigenvalues of Orr-Sommerfeld operator (compare p38.jl)
function p40()
    R = 5772
    clf()
    for N = 40:20:100
        # 2nd- and 4th-order differentiation matrices:
        D, x = cheb(N)
        D² = D^2
        D² = D²[2:N, 2:N]
        S = diagm([0; 1 ./ (1 .- x[2:N] .^ 2); 0])
        D4 = (diagm(1 .- x .^ 2) * D^4 - 8diagm(x) * D^3 - 12D^2) * S
        D4 = D4[2:N, 2:N]

        # Orr-Sommerfeld operators A,B and generalized eigenvalues:
        A = (D4 - 2D² + I) / R - 2im * I - 1im * diagm(1 .- x[2:N] .^ 2) * (D² - I)
        B = D² - I
        ee = eigvals(A, B)
        i = N ÷ 20 - 1
        subplot(2, 2, i)
        plot(real(ee), imag(ee), ".", markersize=8)
        grid(true)
        axis("square")
        axis([-0.8, 0.2, -1, 0])
        title("N = $N    \$\\lambda_{max}\$ = $(round(maximum(real(ee)),sigdigits=7))")
    end
    return gcf()
end
