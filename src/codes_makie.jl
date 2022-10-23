# p1 - convergence of fourth-order finite differences
function p1(Nvec = @. 2^(3:12))
    fig = Figure()
    Axis(
        fig[1, 1],
        xscale=log10, yscale=log10,
        xlabel="N", ylabel="error",
        title="Convergence of fourth-order finite differences",
    )
    # For various N, set up grid in [-π,pi] and function u(x):
    for N in Nvec
        h = 2π / N
        x = @. -π + (1:N) * h
        u = @. exp(sin(x)^2)
        uprime = @. 2 * sin(x) * cos(x) * u

        # Construct sparse fourth-order differentiation matrix:
        col1 = [ 0; -2/3h; 1/12h; zeros(N-5); -1/12h; 2/3h ]
        D = sparse( [col1[mod(i-j,N) + 1] for i in 1:N, j in 1:N] )

        # Plot max(abs(D*u-uprime)):
        error = norm(D * u - uprime, Inf)
        scatter!(N, error, color=:black)
    end
    lines!(Nvec, Nvec .^ (-4), linestyle=:dash)
    text!(105, 5e-8, text=L"N^{-4}")
    return fig
end

# p2 - convergence of periodic spectral method (compare p1.jl)
function p2(Nvec = 2:2:100)
    fig = Figure()
    Axis(
        fig[1, 1],
        xscale=log10, yscale=log10,
        xlabel="N", ylabel="error",
        title="Convergence of spectral differentiation",
    )
    @assert(all(iseven.(Nvec)),"N must be even")
    # For various N (even), set up grid as before:
     for N in Nvec
        h = 2π / N
        x = [-π + i * h for i = 1:N]
        u = @. exp(sin(x))
        uprime = @. cos(x) * u

        # Construct spectral differentiation matrix:
        entry(k) = k==0 ? 0.0 : (-1)^k * 0.5cot( k * h / 2 )
        D = [ entry(mod(i-j,N)) for i in 1:N, j in 1:N ]

        # Plot max(abs(D*u-uprime)):
        error = norm(D * u - uprime, Inf)
        scatter!(N, error, color=:black)
    end
    return fig
end

# p3 - band-limited interpolation
function p3(h = 1)
    xmax = 10
    fig = Figure()
    x = -xmax:h:xmax                     # computational grid
    xx = -xmax-h/20:h/10:xmax+h/20       # plotting grid
    funs = [
        (x -> float(x == 0), "discrete delta"),
        (x -> float(abs(x) <= 3), "square wave"),
        (x -> max(0, 1 - abs(x) / 3), "tent function")
    ]
    for (plt, (u,label)) in enumerate(funs)
        ax = Axis(
            fig[plt, 1],
            xticksvisible=false, xticklabelsvisible=false,
            yticks=0:1
        )
        (plt==1) && (ax.title = "Convergence of spectral differentiation")
        v = u.(x)
        scatter!(x, v)
        p = [ sum(v[i] * sinc((ξ - x[i]) / h) for i in eachindex(x)) for ξ in xx ]
        lines!(xx, p)
        limits!(ax, -xmax, xmax, -0.5, 1.5)
        text!(-xmax+2, 1.05, text=label)
    end
    return fig
end

# p3 - Gibbs phenomenon
function p3g(xmax = 6)
    fig = Figure()
    for (plt, h) in enumerate([1, 1/2, 1/8])
        ax = Axis(
            fig[plt, 1],
            xticksvisible=false, xticklabelsvisible=false,
            yticks=0:1
        )
        (plt==1) && (ax.title = "Gibbs phenomenon")
        x = -xmax:h:xmax                     # computational grid
        v = @. float(abs(x) ≤ 3)
        scatter!(x, v, markersize=9)
        xx = -xmax-h/20:h/10:xmax+h/20       # plotting grid
        p = [ sum(v[i] * sinc((ξ - x[i]) / h) for i in eachindex(x)) for ξ in xx ]
        lines!(xx, p)
        limits!(ax, -xmax, xmax, -0.25, 1.25)
    end
    return fig
end


# p4 - periodic spectral differentiation
function p4(N = 24)
    # Set up grid and differentiation matrix:
    h = 2π / N
    x = h * (1:N)
    entry(k) = k==0 ? 0.0 : 0.5 * (-1)^k * cot(k * h / 2)
    D = [ entry(mod(i-j,N)) for i in 1:N, j in 1:N ]

    # Differentiation of a hat function:
    v = @. max(0, 1 - abs(x - π) / 2)
    fig = Figure()
    ax = Axis(fig[1, 1], title="function")
    scatterlines!(x, v)
    limits!(ax, 0, 2π, -0.5, 1.5)
    ax = Axis(fig[1, 2], title="spectral derivative")
    scatterlines!(x, D * v)
    limits!(ax, 0, 2π, -1, 1)

    # Differentiation of exp(sin(x)):
    v = @. exp(sin(x))
    vʹ = @. cos(x) * v
    ax = Axis(fig[2, 1])
    scatterlines!(x, v)
    limits!(ax, 0, 2π, 0, 3)
    ax = Axis(fig[2, 2])
    scatterlines!(x, D * v)
    limits!(ax, 0, 2π, -2, 2)
    error = norm(D * v - vʹ, Inf)
    text!(2.2, 1.4, text=f"max error = {error:.5g}", textsize=20)
    return fig
end

# p5 - repetition of p4 via FFT
function p5(N = 24)
    function fderiv(v::Vector{T}) where T <: Real
        v̂ = rfft(v)
        ŵ = 1im * [0:N/2-1; 0] .* v̂
        return irfft(ŵ, N) 
    end
    function fderiv(v)
        v̂ = fft(v)
        ŵ = 1im * [0:N/2-1; 0; -N/2+1:-1] .* v̂
        return ifft(ŵ)
    end

    # Differentiation of a hat function:
    h = 2π / N
    x = h * (1:N)
    v = @. max(0, 1 - abs(x - π) / 2)
    w = fderiv(v)
    fig = Figure()
    ax = Axis(fig[1, 1],
        xticks = MultiplesTicks(5, π, "π"), 
        title="function"
    )
    scatterlines!(x, v)
    
    ax = Axis(fig[1, 2], 
        xticks = MultiplesTicks(5, π, "π"),
        title="spectral derivative"
    )
    scatterlines!(x, w)

    # Differentiation of exp(sin(x)):
    v = @. exp(sin(x))
    vʹ = @. cos(x) * v
    w = fderiv(v)
    ax = Axis(fig[2, 1], xticks = MultiplesTicks(5, π, "π"))
    scatterlines!(x, v)
    limits!(ax, 0, 2π, 0, 3)
    ax = Axis(fig[2, 2], xticks = MultiplesTicks(5, π, "π"))
    scatterlines!(x, w)
    limits!(ax, 0, 2π, -2, 2)
    error = norm(w - vʹ, Inf)
    text!(2.2, 1.4, text=f"max error = {error:.5g}", textsize=20)
    return fig
end


# p6 - variable coefficient wave equation
#  use ⍺ = 1.9 to see instability
function p6(⍺ = 1.5)
    # Grid, variable coefficient, and initial data:
    N = 128;  h = 2π / N
    x = h * (1:N)
    t = 0;  Δt = ⍺ / N
    c = @. 0.2 + sin(x - 1)^2
    v = @. exp(-100 * (x - 1) .^ 2)
    vold = @. exp(-100 * (x - 0.2Δt - 1) .^ 2)

    # Time-stepping by leap frog formula:
    tmax = 8
    nsteps = ceil(Int, tmax / Δt)
    Δt = tmax / nsteps
    V = [v fill(NaN, N, nsteps)]
    t = Δt*(0:nsteps)
    for i in 1:nsteps
        v̂ = rfft( V[:,i] )
        ŵ = 1im * [0:N/2-1; 0] .* v̂
        w = irfft(ŵ, N)
        V[:,i+1] = vold - 2Δt * c .* w
        vold = V[:,i]
        if norm(V[:,i+1], Inf) > 2.5
            break 
        end
    end

    fig = Figure()
    Axis3(fig[1, 1],
        xticks = MultiplesTicks(5, π, "π"),
        xlabel="x", ylabel="t", zlabel="u", 
        azimuth=4.5, elevation=1.44,
    )
    gap = max(1,round(Int, 0.15/Δt) - 1)
    surface!(x, t, V, colorrange=(0,1))
    [ lines!(x, fill(t[j], N), V[:, j].+.01, color=:ivory) for j in 1:gap:nsteps+1 ]
    return fig
end

# p7 - accuracy of periodic spectral differentiation
function p7(allN = 6:2:50)
    @assert(all(iseven.(allN)),"N must be even")
    # Compute derivatives for various values of N:
    Nmax = maximum(allN)
    data = [ 
        # uʹʹʹ in BV
        (x -> abs(sin(x))^3,  x -> 3 * sin(x) * cos(x) * abs(sin(x)), 
            L"|\sin(x)|^3", (1,1)), 
        # C-infinity
        (x -> exp(-sin(x / 2)^(-2)), 
            x -> 0.5exp(-sin(x / 2)^(-2)) * sin(x) / sin(x / 2)^4, 
            L"\exp(-\sin^{-2}(x/2))", (1,2)), 
        # analytic in a strip
        (x -> 1 / (1 + sin(x / 2)^2), 
            x -> -sin(x / 2) * cos(x / 2) / (1 + sin(x / 2)^2)^2,
            L"1/(1+\sin^2(x/2))", (2,1) ),
        # band-limited 
        (x -> sin(10x), x -> 10cos(10x),  L"\sin(10x)", (2,2))
    ]
    fig = Figure()
    E = zeros(length(allN))
    ax = []
    for (fun,deriv,title,pos) in data
        for (k,N) in enumerate(allN)
            h = 2π / N
            x = h * (1:N)
            entry(k) = k==0 ? 0.0 : 0.5 * (-1)^k * cot(k * h / 2)
            D = [ entry(mod(i-j, N)) for i in 1:N, j in 1:N ]
            E[k] = norm(D * fun.(x) - deriv.(x), Inf)
        end
        push!(ax,
            Axis(fig[pos[1], pos[2]],
                title=title, yscale=log10
            ))
        scatterlines!(allN, E)
        ax[end].xlabel = (pos[1] == 2) ? "N" : ""
        ax[end].ylabel = (pos[2] == 1) ? "error" : ""
    end
    linkxaxes!(ax...)
    linkyaxes!(ax...)
    return fig
end

# p8 - eigenvalues of harmonic oscillator -u"+x^2 u on R
function p8( N = 6:6:36)
    L = 8                             # domain is [-L L], periodic
    λ = zeros(4,0)
    for N in N
        h = 2π / N
        x = [ (L/π)*(i*h - π) for i in 1:N ]
        entry(k) = k==0 ? -π^2 / 3h^2 - 1/6 : -0.5 * (-1)^k / sin(h * k / 2)^2
        D² = [(π / L)^2 * entry(mod(i-j, N)) for i in 1:N, j in 1:N]  # 2nd-order differentiation
        λ = [ λ eigvals(-D² + diagm(x .^ 2))[1:4] ]
    end
    header = ["N = $n" for n in N]
    pretty_table(λ; header, formatters=ft_printf("%.14f"))
end

# p9 - polynomial interpolation in equispaced and Chebyshev pts
function p9(N = 16)
    xx = -1.01:0.005:1.01
    fig = Figure()
    labels = ["equispaced points", "Chebyshev points"]
    points = [-1 .+ 2 * (0:N) / N, cospi.((0:N) / N)]
    for (i,(s,x)) in enumerate(zip(labels,points))
        ax = Axis(fig[i, 1], title=s)
        u = @. 1 / (1 + 16 * x^2)
        uu = @. 1 / (1 + 16 * xx^2)
        p = polyinterp(x, u)              # interpolation
        pp = p.(xx)                       # evaluation of interpolant
        lines!(xx, pp)
        scatter!(x, u)
        limits!(-1.05, 1.05, -1, 1.5)
        error = norm(uu - pp, Inf)
        text!(-0.5, -0.5, text=f"max error = {error:.5g}")
    end
    return fig
end

# p10 - polynomials and corresponding equipotential curves
function p10(N = 16)
    fig = Figure()
    xx = -1.01:0.005:1.01
    labels = ["equispaced points", "Chebyshev points"]
    points = [-1 .+ 2 * (0:N) / N, cospi.((0:N) / N)]
    for (i,(s,x)) in enumerate(zip(labels,points))
        p = fromroots(x)

        # Plot p(x) over [-1,1]:
        xx = -1:0.005:1
        pp = p.(xx)
        Axis(fig[i, 1], xticks=-1:0.5:1, title=s)
        scatter!(x, zero(x))
        lines!(xx, pp)

        # Plot equipotential curves:
        Axis(fig[i, 2], xticks=-1:0.5:1, title=s)
        scatter!(real(x), imag(x))
        xx = -1.4:0.02:1.4
        yy = -1.12:0.02:1.12
        zz = [complex(x,y) for x in xx, y in yy]
        pp = p.(zz)
        levels = 10.0 .^ (-4:0)
        contour!(xx, yy, abs.(pp); levels, color=:black)
        limits!(-1.4, 1.4, -1.12, 1.12)
    end
    return fig
end

# p11 - Chebyshev differentation of a smooth function
function p11()
    u = x -> exp(x) * sin(5x) 
    uʹ = x -> exp(x) * (sin(5x) + 5 * cos(5x))
    xx = (-200:200) / 200
    vv = @. u.(xx)
    fig = Figure()
    for (i,N) in enumerate([10, 20])
        D, x = cheb(N)
        v = u.(x)
        Axis(fig[i, 1], title="u(x),  N=$N")
        scatter!(x, v)
        lines!(xx, vv)
        error = D * v - uʹ.(x)
        Axis(fig[i, 2], title="error in uʹ(x),  N=$N")
        scatter!(x, error)
        lines!(xx, polyinterp(x, error).(xx))
    end
    return fig
end

# p12 - accuracy of Chebyshev spectral differentiation
function p12(Nmax = 50)
    # Compute derivatives for various values of N:
    data = [ 
        # uʹʹʹ in BV
        (x -> abs(x)^3,  x -> 3x * abs(x), L"|x|^3", (1,1)), 
        # C-infinity
        (x -> exp(-x^(-2)), x -> 2exp(-x^(-2)) / x^3, L"\exp(-x^{-2})", (1,2)), 
        # analytic in [-1,1]
        (x -> 1 / (1 + x^2), x -> -2x / (1 + x^2)^2, L"1/(1+x^2)", (2,1) ),
        # polynomial 
        (x -> x^10, x -> 10x^9,  L"x^{10}", (2,2))
    ]
    fig = Figure()
    E = zeros(Nmax)
    ax = []
    for (u,uʹ,title,pos) in data
        for N in 1:Nmax
            D, x = cheb(N)
            E[N] = norm(D * u.(x) - uʹ.(x), Inf)
        end
        push!(
            ax,
            Axis(fig[pos[1], pos[2]];title, yscale=log10)
        )
        scatterlines!(1:Nmax, E)
        (pos[1] == 2) && (ax[end].xlabel = "N")
        (pos[2] == 1) && (ax[end].ylabel = "error")
    end
    linkxaxes!(ax...)
    linkyaxes!(ax...)
    return fig
end

# p13 - solve linear BVP u_xx = exp(4x), u(-1)=u(1)=0
function p13(N = 16)
    D, x = cheb(N)
    D² = (D^2)[2:N, 2:N]                   # boundary conditions
    f = @. exp(4x[2:N])
    u = D² \ f                           # Poisson eq. solved here
    u = [0; u; 0]
    xx = -1:0.01:1
    uu = polyinterp(x, u).(xx)      # interpolate grid data
    exact = @. (exp(4xx) - sinh(4) * xx - cosh(4)) / 16
    err = norm(uu - exact,Inf)
    fig = Figure()
    Axis( fig[1, 1], title=f"max err = {norm(uu-exact,Inf):.4g}" )
    scatter!(x, u)
    lines!(xx, uu)
    return fig
end

# p14 - solve nonlinear BVP u_xx = exp(u), u(-1)=u(1)=0
function p14(N = 16)
    D, x = cheb(N)
    D² = (D^2)[2:N, 2:N]
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
    xx = -1:0.01:1
    uu = polyinterp(x,u).(xx)
    fig = Figure()
    Axis( fig[1, 1], title="no. steps = $it      u(0) = $(u[N÷2+1])" )
    scatter!(x, u)
    lines!(xx, uu)
    return fig
end

# p15 - solve eigenvalue BVP u_xx = lambda*u, u(-1)=u(1)=0
function p15(N = 36)
    D, x = cheb(N)
    D² = (D^2)[2:N, 2:N]
    λ, V = eigen(D², sortby = (-)∘real)
    fig = Figure()
    xx = -1:0.01:1
    for j in 5:5:30                  # plot 6 eigenvectors
        u = [0; V[:, j]; 0]
        ax = Axis( fig[j÷5, 1] )
        scatter!(x, u)
        uu = polyinterp(x,u).(xx)
        lines!(xx, uu)
        hidespines!(ax); hidedecorations!(ax)
        text!(-0.4, 0.12, text=f"eig {j} = {λ[j]*4/π^2:#.14g} π^2/4", textsize=20)
        text!(0.7, 0.12, text=f"{4*N/(π*j):.2g} ppw", textsize=20)
    end
    return fig
end

# p16 - Poisson eq. on [-1,1]x[-1,1] with u=0 on boundary
function p16(N = 24)
    # Set up grids and tensor product Laplacian and solve for u:
    ⊗ = kron
    D, x = D, y = cheb(N)
    F = [ 10sin(8x * (y - 1)) for x in x[2:N], y in y[2:N] ]
    D² = (D^2)[2:N, 2:N]
    L = I(N-1) ⊗ D² + D² ⊗ I(N-1)                     # Laplacian

    fig = Figure()
    # Axis(fig[1, 1], title="Matrix nonzeros", aspect=DataAspect())
    # spy!(sparse(L), markersize=4)
    # ylims!((N-1)^2, 1)

    @elapsed u = L \ vec(F)           # solve problem and watch the clock
    
    # Reshape long 1D results onto 2D grid (flipping orientation):
    U = zeros(N+1, N+1)
    U[2:N, 2:N] = reshape(u, N-1, N-1)
    value = U[N÷4 + 1, N÷4 + 1]

    # Interpolate to finer grid and plot:
    xx = yy = -1:0.04:1
    UU = gridinterp(U,xx,yy)
 
    ax3 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="u")
    surface!(xx, yy, UU)
    ax3.azimuth = 6π / 5; ax3.elevation = π / 6
    val = f"{value:.11g}"
    text!(0.4, -0.3, 0.3, text=latexstring("u(2^{-1/2},\\,2^{-1/2}) = "*val))
    return fig
end

# p17 - Helmholtz eq. u_xx + u_yy + (k^2)u = f
function p17(N = 24)
    # Set up spectral grid and tensor product Helmholtz operator:
    ⊗ = kron
    D, x = D, y = cheb(N)
    F = [exp(-10 * ((y - 1)^2 + (x - 0.5)^2)) for x in x[2:N], y in y[2:N]]
    D² = (D^2)[2:N, 2:N]
    k = 9
    L = I(N-1) ⊗ D² + D² ⊗ I(N-1) + k^2 * I

    # Solve for u, reshape to 2D grid, and plot:
    u = L \ vec(F)
    U = zeros(N+1, N+1)
    U[2:N, 2:N] = reshape(u, N-1, N-1)
    xx = yy = -1:1/50:1
    UU = gridinterp(U, xx, yy)
    value = U[N÷2 + 1, N÷2 + 1]
 
    fig = Figure()
    Axis(
        fig[1, 1], 
        aspect = DataAspect(), xlabel="x", ylabel="y", 
        title = f"u(0,0) = {value:.10f}"
    )
    co = contourf!(xx, yy, UU)
    Colorbar(fig[1,2], co)
    return fig
end

# p18 - Chebyshev differentiation via FFT (compare p11.jl)
function p18()
    xx = -1:0.01:1
    ff = @. exp(xx) * sin(5 * xx)
    clf()
    for N = [10 20]
        x = cheb(N)[2]
        f = @. exp(x) * sin(5 * x)
        PyPlot.axes([0.15, 0.66 - 0.4 * (N == 20), 0.31, 0.28])
        plot(x, f, "k.", markersize=6)
        grid(true)
        plot(xx, ff)
        title("f(x), N=$N")
        error = chebfft(f) - @. exp(x) * (sin(5 * x) + 5 * cos(5 * x))
        PyPlot.axes([0.55, 0.66 - 0.4 * (N == 20), 0.31, 0.28])
        plot(x, error, ".-", markersize=10)
        grid(true)
        title("error in f'(x),  N=$N")
    end
end

# p19 - 2nd-order wave eq. on Chebyshev grid (compare p6.jl)
function p19()
    # Time-stepping by leap frog formula:
    N = 80
    x = cheb(N)[2]
    dt = 8 / N^2
    v = @. exp(-200 * x^2)
    vold = @. exp(-200 * (x - dt)^2)
    tmax = 4
    tplot = 0.075
    plotgap = round(Int, tplot / dt)
    dt = tplot / plotgap
    nplots = round(Int, tmax / tplot)
    plotdata = [v zeros(N + 1, nplots)]
    tdata = 0
    for i = 1:nplots
        global v, vold
        global plotdata, tdata
        for n = 1:plotgap
            w = chebfft(chebfft(v))
            w[1] = 0
            w[N+1] = 0
            vnew = 2 * v - vold + dt^2 * w
            vold = v
            v = vnew
        end
        plotdata[:, i+1] = v
        tdata = [tdata; dt * i * plotgap]
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
end

# p20 - 2nd-order wave eq. in 2D via FFT (compare p19.m)
function p20()
    # Grid and initial data:
    N = 24
    x = y = cheb(N)[2]
    dt = 6 / N^2
    xx = yy = x[end:-1:1]
    plotgap = round(Int, (1 / 3) / dt)
    dt = (1 / 3) / plotgap
    vv = @. exp(-40 * ((x' - 0.4)^2 + y^2))
    vvold = vv
    clf()

    # Time-stepping by leap frog formula:
    for n = 0:3*plotgap
        t = n * dt
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
            W1 = real(ifft(1im * [0:N-1; 0; 1-N:-1] .* U)) # diff wrt theta
            W2 = real(ifft(-[0:N; 1-N:-1] .^ 2 .* U))     # diff^2 wrt theta
            uxx[i, ii] = W2[ii] ./ (1 .- x[ii] .^ 2) - x[ii] .* W1[ii] ./ (1 .- x[ii] .^ 2) .^ (3 / 2)
        end
        for j = 2:N                # 2nd derivs wrt y in each column
            v = vv[:, j]
            V = [v; reverse(v[ii])]
            U = real(fft(V))
            W1 = real(ifft(1im * [0:N-1; 0; 1-N:-1] .* U))# diff wrt theta
            W2 = real(ifft(-[0:N; 1-N:-1] .^ 2 .* U))    # diff^2 wrt theta
            uyy[ii, j] = W2[ii] ./ (1 .- y[ii] .^ 2) - y[ii] .* W1[ii] ./ (1 .- y[ii] .^ 2) .^ (3 / 2)
        end
        vvnew = 2 * vv - vvold + dt^2 * (uxx + uyy)
        vvold = vv
        vv = vvnew
    end
end

# p21 - eigenvalues of Mathieu operator -u_xx + 2qcos(2x)u
function p21()
    N = 42
    h = 2 * π / N
    x = h * (1:N)
    D² = toeplitz([-π^2 / (3 * h^2) - 1 / 6; @. -0.5 * (-1)^(1:N-1) / sin(h * (1:N-1) / 2)^2])
    qq = 0:0.2:15
    data = zeros(0, 11)
    for q = qq
        e = sort(eigvals(-D² + 2 * q * diagm(cos.(2 * x))))
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
end

# p22 - 5th eigenvector of Airy equation u_xx = lambda*x*u
function p22()
    clf()
    for N = 12:12:48
        D, x = cheb(N)
        D² = D^2
        D² = D²[2:N, 2:N]
        lam, V = eigen(D², diagm(x[2:N]))      # generalized ev problem
        ii = findall(lam .> 0)
        V = V[:, ii]
        lam = lam[ii]
        ii = sortperm(lam)[5]
        lambda = lam[ii]
        v = [0; V[:, ii]; 0]
        v = v / v[Int(N / 2 + 1)] * airyai(0)
        xx = -1:0.01:1
        vv = fit(x, v).(xx)
        subplot(2, 2, N ÷ 12)
        plot(xx, vv)
        grid(true)
        title("N = $N     eig = $(round(lambda,sigdigits=13))")
    end
end

# p23 - eigenvalues of perturbed Laplacian on [-1,1]x[-1,1]
function p23()
    # Set up tensor product Laplacian and compute 4 eigenmodes:
    N = 16
    (D, x) = cheb(N)
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
        title("eig = $(round(D[i]/(pi^2/4),sigdigits=13)) π^2/4")
    end
end

# p23a - eigenvalues of UNperturbed Laplacian on [-1,1]x[-1,1]
function p23a()
    # Set up tensor product Laplacian and compute 4 eigenmodes:
    N = 16
    (D, x) = cheb(N)
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
        title("eig = $(round(D[i]/(pi^2/4),sigdigits=13)) π^2/4")
    end
end

# p24 - pseudospectra of Davies's complex harmonic oscillator
function p24()
    # Eigenvalues:
    N = 70
    (D, x) = cheb(N)
    x = x[2:N]
    L = 6
    x = L * x
    D = D / L                   # rescale to [-L,L]
    A = -D^2
    A = A[2:N, 2:N] + (1 + 3im) * diagm(x .^ 2)
    lambda = eigvals(A)
    clf()
    plot(real(lambda), imag(lambda), "k.", markersize=6)
    axis([0, 50, 0, 40])

    # Pseudospectra:
    x = 0:1:50
    y = 0:1:40
    zz = x' .+ 1im * y
    minsvd(z) = minimum(svdvals(z * I - A))
    sigmin = [minsvd(x[i] + 1im * y[j]) for i = eachindex(x), j = eachindex(y)]
    contour(x, y, sigmin', levels=10.0 .^ (-4:0.5:-0.5))
end

# p24 - pseudospectra of Davies's complex harmonic oscillator
function p24fine()
    # Eigenvalues:
    N = 70
    (D, x) = cheb(N)
    x = x[2:N]
    L = 6
    x = L * x
    D = D / L                   # rescale to [-L,L]
    A = -D^2
    A = A[2:N, 2:N] + (1 + 3im) * diagm(x .^ 2)
    lambda = eigvals(A)
    clf()
    plot(real(lambda), imag(lambda), "k.", markersize=6)
    axis([0, 50, 0, 40])

    # Pseudospectra:
    x = 0:0.5:50
    y = 0:0.5:40
    minsvd(z) = minimum(svdvals(z * I - A))
    sigmin = [minsvd(x[i] + 1im * y[j]) for i = eachindex(x), j = eachindex(y)]
    contour(x, y, sigmin', levels=10.0 .^ (-4:0.5:-0.5))
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
    s = @. (5 * z + 8 - 1 / z) / 12
    zplot(r ./ s)                    # order 3
    s = @. (9 * z + 19 - 5 / z + 1 / z^2) / 24
    zplot(r ./ s)           # order 4
    s = @. (251 * z + 646 - 264 / z + 106 / z^2 - 19 / z^3) / 720
    zplot(r ./ s)   # 5
    d = @. 1 - 1 / z
    s = @. 1 - d / 2 - d^2 / 12 - d^3 / 24 - 19 * d^4 / 720 - 3 * d^5 / 160
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
        w[2] -= (1 + w[2] + 0.5 * w[2]^2 - z[i]^2) / (1 + w[2])
        w[3] -= (1 + w[3] + 0.5 * w[3]^2 + w[3]^3 / 6 - z[i]^3) / (1 + w[3] + w[3]^2 / 2)
        w[4] -= (1 + w[4] + 0.5 * w[4]^2 + w[4]^3 / 6 + w[4] .^ 4 / 24 - z[i]^4) / (1 + w[4] + w[4]^2 / 2 + w[4]^3 / 6)
        W = [W; transpose(w)]
    end
    zplot(W)
    axis("square")
    axis([-5, 2, -3.5, 3.5])
    grid(true)
    title("Runge—Kutta")
end

# p26 - eigenvalues of 2nd-order Chebyshev diff. matrix
function p26()
    N = 60
    (D, x) = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]
    (lam, V) = eigen(D²)
    ii = sortperm(-lam)
    e = lam[ii]
    V = V[:, ii]

    # Plot eigenvalues:
    clf()
    PyPlot.axes([0.1, 0.62, 0.8, 0.3])
    loglog(-e, ".", markersize=4)
    ylabel("eigenvalue")
    title("N = $N       max |λ| = $(round(maximum(-e)/N^4,sigdigits=5)) \$N^4\$")
    semilogy(2 * N / pi * [1, 1], [1, 1e6], "--r")
    text(2.1 * N / pi, 24, "2π / N", fontsize=12)

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
end

# p27 - Solve KdV eq. u_t + uu_x + u_xxx = 0 on [-π,pi] by
function p27()
    # Set up grid and two-soliton initial data:
    N = 256
    dt = 0.4 / N^2
    x = (2 * π / N) * (-N/2:N/2-1)
    A = 25
    B = 16
    clf()
    u = @. 3 * A^2 * sech(0.5 * (A * (x + 2)))^2 + 3 * B^2 * sech(0.5 * (B * (x + 1)))^2
    v = fft(u)
    k = [0:N/2-1; 0; -N/2+1:-1]
    ik3 = 1im * k .^ 3

    # Solve PDE and plot results:
    tmax = 0.006
    nplt = floor(Int, (tmax / 25) / dt)
    nmax = round(Int, tmax / dt)
    udata = u
    tdata = [0.0]
    for n = 1:nmax
        t = n * dt
        g = -0.5im * dt * k
        E = exp.(dt * ik3 / 2)
        E2 = E .^ 2
        a = g .* fft(real(ifft(v)) .^ 2)
        b = g .* fft(real(ifft(E .* (v + a / 2))) .^ 2)     # 4th-order
        c = g .* fft(real(ifft(E .* v + b / 2)) .^ 2)     # Runge-Kutta
        d = g .* fft(real(ifft(E2 .* v + E .* c)) .^ 2)
        v = E2 .* v + (E2 .* a + 2 * E .* (b + c) + d) / 6
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
    xlim(-π, pi)
    ylim(0, tmax)
    zlim(0, 12000)
    |
    zticks([0, 2000])
end

# p28 - eigenmodes of Laplacian on the disk (compare p22.jl)
function p28()
    # r coordinate, ranging from -1 to 1 (N must be odd):
    N = 25
    N2 = Int((N - 1) / 2)
    (D, r) = cheb(N)
    D² = D^2
    D1 = D²[2:N2+1, 2:N2+1]
    D² = D²[2:N2+1, N:-1:N2+2]
    E1 = D[2:N2+1, 2:N2+1]
    E2 = D[2:N2+1, N:-1:N2+2]

    # t = theta coordinate, ranging from 0 to 2*pi (M must be even):
    M = 20
    dt = 2 * π / M
    t = dt * (1:M)
    M2 = Int(M / 2)
    D²t = toeplitz([-π^2 / (3 * dt^2) - 1 / 6; @. 0.5 * (-1)^(2:M) / sin(dt * (1:M-1) / 2)^2])

    # Laplacian in polar coordinates:
    R = diagm(1 ./ r[2:N2+1])
    Z = zeros(M2, M2)
    L = kron(D1 + R * E1, I(M)) + kron(D² + R * E2, [Z I(M2); I(M2) Z]) + kron(R^2, D²t)

    # Compute four eigenmodes:
    index = [1, 3, 6, 10]
    Lam, V = eigen(-L)
    ii = sortperm(abs.(Lam))[index]
    Lam = Lam[ii]
    V = V[:, ii]
    Lam = sqrt.(real(Lam / Lam[1]))

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
        title("Mode $(index[i]):  λ = $(round(Lam[i],sigdigits=11))", fontsize=9)
    end
end

# p28b - eigenmodes of Laplacian on the disk
function p28b()
    # r coordinate, ranging from -1 to 1 (N must be odd)
    N = 25
    N2 = Int((N - 1) / 2)
    (D, r) = cheb(N)
    D² = D^2
    D1 = D²[2:N2+1, 2:N2+1]
    D² = D²[2:N2+1, N:-1:N2+2]
    E1 = D[2:N2+1, 2:N2+1]
    E2 = D[2:N2+1, N:-1:N2+2]

    # t = theta coordinate, ranging from 0 to 2*pi (M must be even):
    M = 20
    dt = 2 * π / M
    t = dt * (1:M)
    M2 = Int(M / 2)
    D²t = toeplitz([-π^2 / (3 * dt^2) - 1 / 6; @. 0.5 * (-1)^(2:M) / sin(dt * (1:M-1) / 2)^2])

    # Laplacian in polar coordinates:
    R = diagm(1 ./ r[2:N2+1])
    Z = zeros(M2, M2)
    L = kron(D1 + R * E1, I(M)) + kron(D² + R * E2, [Z I(M2); I(M2) Z]) + kron(R^2, D²t)

    # Compute 25 eigenmodes:
    index = 1:25
    Lam, V = eigen(-L)
    ii = sortperm(abs.(Lam))[index]
    Lam = Lam[ii]
    V = V[:, ii]
    Lam = sqrt.(real(Lam / Lam[1]))

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
        title("$(round(Lam[i],sigdigits=5))", fontsize=8)
    end
end

# p29 - solve Poisson equation on the unit disk
function p29()
    # Laplacian in polar coordinates:
    N = 25
    N2 = Int((N - 1) / 2)
    (D, r) = cheb(N)
    D² = D^2
    D1 = D²[2:N2+1, 2:N2+1]
    D² = D²[2:N2+1, N:-1:N2+2]
    E1 = D[2:N2+1, 2:N2+1]
    E2 = D[2:N2+1, N:-1:N2+2]
    M = 20
    dt = 2 * π / M
    t = dt * (1:M)
    M2 = Int(M / 2)
    D²t = toeplitz([-π^2 / (3 * dt^2) - 1 / 6; @. 0.5 * (-1)^(2:M) / sin(dt * (1:M-1) / 2)^2])
    R = diagm(1 ./ r[2:N2+1])
    Z = zeros(M2, M2)
    L = kron(D1 + R * E1, I(M)) + kron(D² + R * E2, [Z I(M2); I(M2) Z]) + kron(R^2, D²t)

    # Right-hand side and solution for u:
    (rr, tt) = (r[2:N2+1]', t)
    f = @. -rr^2 * sin(tt / 2)^4 + sin(6 * tt) * cos(tt / 2)^2
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
end

# p30 - spectral integration, ODE style (compare p12.jl)
function p30()
    Nmax = 50
    # Computation: various values of N, four functions:
    clf()
    for N = 1:Nmax
        i = 1:N
        (D, x) = cheb(N)
        x = x[i]
        Di = inv(D[i, i])
        w = Di[1, :]
        f = @. abs(x)^3
        E[1, N] = abs(dot(w, f) - 0.5)
        f = @. exp(-x^(-2))
        E[2, N] = abs(dot(w, f) - 2 * (exp(-1) + sqrt(pi) * (erf(1) - 1)))
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
end

# p30b - spectral integration, Clenshaw-Curtis style (compare p30.jl)
function p30b()
    Nmax = 50
    # Computation: various values of N, four functions:
    clf()
    for N = 1:Nmax
        (x, w) = clencurt(N)
        f = @. abs(x)^3
        E[1, N] = abs(dot(w, f) - 0.5)
        f = @. exp(-x^(-2))
        E[2, N] = abs(dot(w, f) - 2 * (exp(-1) + sqrt(pi) * (erf(1) - 1)))
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
end

# p30c - spectral integration, Gauss style (compare p30.jl)
function p30c()
    Nmax = 50
    # Computation: various values of N, four functions:
    clf()
    for N = 1:Nmax
        (x, w) = gauss(N)
        f = @. abs(x)^3
        E[1, N] = abs(dot(w, f) - 0.5)
        f = @. exp(-x^(-2))
        E[2, N] = abs(dot(w, f) - 2 * (exp(-1) + sqrt(pi) * (erf(1) - 1)))
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
end

# p31 - gamma function via complex integral, trapezoid rule
function p31()
    N = 70
    theta = @. -π + (2 * π / N) * (0.5:N-0.5)
    c = -11                     # center of circle of integration
    r = 16                      # radius of circle of integration
    x = -3.5:0.1:4
    y = -2.5:0.1:2.5
    zz = x' .+ 1im * y
    gaminv = 0 * zz
    for i = 1:N
        t = c + r * exp(1im * theta[i])
        gaminv += exp(t) * t .^ (-zz) * (t - c)
    end
    gaminv = gaminv / N
    gam = 1 ./ gaminv
    clf()
    surf(x, y, abs.(gam))
    xlim(-3.5, 4)
    ylim(-2.5, 2.5)
    zlim(0, 6)
    xlabel("Re(z)")
    ylabel("Im(z)")
    text3D(4, -1.4, 5.5, "\$|\\Gamma(z)|\$", fontsize=20)
end

# p32 - solve u_xx = exp(4x), u(-1)=0, u(1)=1 (compare p13.jl)
function p32()
    N = 16
    (D, x) = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]                   # boundary conditions
    f = @. exp(4 * x[2:N])
    u = D² \ f                           # Poisson eq. solved here
    u = [0; u; 0] + (x .+ 1) / 2
    clf()
    plot(x, u, ".", markersize=10)
    xx = -1:0.01:1
    uu = polyinterp(x,u).(xx)
    plot(xx, uu)
    grid(true)
    exact = @. (exp(4 * xx) - sinh(4) * xx - cosh(4)) / 16 + (xx + 1) / 2
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=4))", fontsize=12)
end

# p33 - solve linear BVP u_xx = exp(4x), u'(-1)=u(1)=0
function p33()
    N = 16
    (D, x) = cheb(N)
    D² = D^2
    D²[N+1, :] = D[N+1, :]            # Neumann condition at x = -1
    D² = D²[2:N+1, 2:N+1]
    f = @. exp(4 * x[2:N])
    u = D² \ [f; 0]
    u = [0; u]
    clf()
    plot(x, u, ".", markersize=10)
    axis([-1, 1, -4, 0])
    xx = -1:0.01:1
    uu = polyinterp(x,u).(xx)
    plot(xx, uu)
    grid(true)
    exact = @. (exp(4 * xx) - 4 * exp(-4) * (xx - 1) - exp(4)) / 16
    title("max err = $(round(norm(uu-exact,Inf),sigdigits=5))", fontsize=12)
end

# p34 - Allen-Cahn eq. u_t = eps*u_xx+u-u^3, u(-1)=-1, u(1)=1
function p34()
    # Differentiation matrix and initial data:
    N = 20
    (D, x) = cheb(N)
    D² = D^2     # use full-size matrix
    D²[[1, N + 1], :] .= 0                     # for convenience
    eps = 0.01
    dt = min(0.01, 50 / (N^4 * eps))
    t = 0.0
    v = @. 0.53 * x + 0.47 * sin(-1.5 * π * x)

    # Solve PDE by Euler formula and plot results:
    tmax = 100
    tplot = 2
    nplots = round(Int, tmax / tplot)
    plotgap = round(Int, tplot / dt)
    dt = tplot / plotgap
    xx = -1:0.025:1
    vv = fit(x, v).(xx)
    plotdata = [vv zeros(length(xx), nplots)]
    tdata = t
    for i = 1:nplots
        for n = 1:plotgap
            t = t + dt
            v = v + dt * (eps * D² * (v - x) + v - v .^ 3)    # Euler
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
end

# p35 - Allen-Cahn eq. as in p34.m, but with boundary condition
function p35()
    # Differentiation matrix and initial data:
    N = 20
    (D, x) = cheb(N)
    D² = D^2     # use full-size matrix
    eps = 0.01
    dt = min(0.01, 50 / (N^4 * eps))
    t = 0.0
    v = @. 0.53 * x + 0.47 * sin(-1.5 * π * x)

    # Solve PDE by Euler formula and plot results:
    tmax = 100
    tplot = 2
    nplots = round(Int, tmax / tplot)
    plotgap = round(Int, tplot / dt)
    dt = tplot / plotgap
    xx = -1:0.025:1
    vv = fit(x, v).(xx)
    plotdata = [vv zeros(length(xx), nplots)]
    tdata = t
    for i = 1:nplots
        for n = 1:plotgap
            t = t + dt
            v = v + dt * (eps * D² * (v - x) + v - v .^ 3)    # Euler
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
end

# p36 - Laplace eq. on [-1,1]x[-1,1] with nonzero BCs
function p36()
    # Set up grid and 2D Laplacian, boundary points included:
    N = 24
    (D, x) = cheb(N)
    y = x
    xx = repeat(x', outer=(N + 1, 1))
    yy = repeat(y, outer=(1, N + 1))
    D² = D^2
    L = kron(I(N + 1), D²) + kron(D², I(N + 1))

    # Impose boundary conditions by replacing appropriate rows of L:
    b = @. (abs(xx[:]) == 1) | (abs(yy[:]) == 1)            # boundary pts
    L[b, :] .= 0
    L[b, b] = I(4 * N)
    rhs = zeros((N + 1)^2)
    rhs[b] = @. (yy[b] == 1) * (xx[b] < 0) * sin(pi * xx[b])^4 + 0.2 * (xx[b] == 1) * sin(3 * π * yy[b])

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
end

# p37 - 2D "wave tank" with Neumann BCs for |y|=1
function p37()
    # x variable in [-A,A], Fourier:
    A = 3
    Nx = 50
    dx = 2 * A / Nx
    x = @. -A + dx * (1:Nx)
    D²x = (pi / A)^2 * toeplitz([-1 / (3 * (dx / A)^2) - 1 / 6
        @. 0.5 * (-1) .^ (2:Nx) / sin((pi * dx / A) * (1:Nx-1) / 2)^2])

    # y variable in [-1,1], Chebyshev:
    Ny = 15
    Dy, y = cheb(Ny)
    D²y = Dy^2
    BC = -Dy[[1, Ny + 1], [1, Ny + 1]] \ Dy[[1, Ny + 1], 2:Ny]

    # Grid and initial data:
    vv = @. exp(-8 * ((x' + 1.5)^2 + y^2))
    dt = 5 / (Nx + Ny^2)
    vvold = @. exp(-8 * ((x' + dt + 1.5) .^ 2 + y .^ 2))

    # Time-stepping by leap frog formula:
    plotgap = round(Int, 2 / dt)
    dt = 2 / plotgap
    for n = 0:2*plotgap
        t = n * dt
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
        vvnew = 2 * vv - vvold + dt^2 * (vv * D²x + D²y * vv)
        vvold = vv
        vv = vvnew
        vv[[1, Ny + 1], :] = BC * vv[2:Ny, :]       # Neumann BCs for |y|=1
    end
end

# p38 - solve u_xxxx = exp(x), u(-1)=u(1)=u'(-1)=u'(1)=0
function p38()
    # Construct discrete biharmonic operator:
    N = 15
    (D, x) = cheb(N)
    S = diagm([0; 1 ./ (1 .- x[2:N] .^ 2); 0])
    D4 = (diagm(1 .- x .^ 2) * D^4 - 8 * diagm(x) * D^3 - 12 * D^2) * S
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
end

# p39 - eigenmodes of biharmonic on a square with clamped BCs
function p39()
    # Construct spectral approximation to biharmonic operator:
    N = 17
    (D, x) = cheb(N)
    D² = D^2
    D² = D²[2:N, 2:N]
    S = diagm([0; 1 ./ (1 .- x[2:N] .^ 2); 0])
    D4 = (diagm(1 .- x .^ 2) * D^4 - 8 * diagm(x) * D^3 - 12 * D^2) * S
    D4 = D4[2:N, 2:N]
    L = kron(I(N - 1), D4) + kron(D4, I(N - 1)) + 2 * kron(D², I(N - 1)) * kron(I(N - 1), D²)

    # Find and plot 25 eigenmodes:
    Lam, V = eigen(-L)
    Lam = -real(Lam)
    ii = sortperm(Lam)[1:25]
    Lam = Lam[ii]
    V = real(V[:, ii])
    Lam = sqrt.(Lam / Lam[1])
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
        text(-0.3, 1.15, "$(round(Lam[i],sigdigits=5))", fontsize=7)
    end
end

# p40 - eigenvalues of Orr-Sommerfeld operator (compare p38.jl)
function p40()
    R = 5772
    clf()
    for N = 40:20:100
        # 2nd- and 4th-order differentiation matrices:
        (D, x) = cheb(N)
        D² = D^2
        D² = D²[2:N, 2:N]
        S = diagm([0; 1 ./ (1 .- x[2:N] .^ 2); 0])
        D4 = (diagm(1 .- x .^ 2) * D^4 - 8 * diagm(x) * D^3 - 12 * D^2) * S
        D4 = D4[2:N, 2:N]

        # Orr-Sommerfeld operators A,B and generalized eigenvalues:
        A = (D4 - 2 * D² + I) / R - 2im * I - 1im * diagm(1 .- x[2:N] .^ 2) * (D² - I)
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
end
