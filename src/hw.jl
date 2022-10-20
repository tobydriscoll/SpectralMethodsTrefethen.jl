function prob4_4()
    # Compute derivatives for various values of N:
    Nmax = 50
    allN = 6:2:Nmax
    data = [ 
        # uʹʹʹ in BV
        (x -> abs(sin(x))^3,  x -> 3 * sin(x) * cos(x) * abs(sin(x)), 
            L"|\sin(x)|^3", (1,1), h -> h^2), 
        # C-infinity
        (x -> exp(-sin(x / 2)^(-2)), 
            x -> 0.5exp(-sin(x / 2)^(-2)) * sin(x) / sin(x / 2)^4, 
            L"\exp(-\sin^{-2}(x/2))", (1,2), h -> NaN), 
        # analytic in a strip
        (x -> 1 / (1 + sin(x / 2)^2), 
            x -> -sin(x / 2) * cos(x / 2) / (1 + sin(x / 2)^2)^2,
            L"1/(1+\sin^2(x/2))", (2,1), h -> exp(-2π*imag(asin(1im))/h) ),
        # band-limited 
        (x -> sin(10x), x -> 10cos(10x),  L"\sin(10x)", (2,2), h -> NaN)
    ]
    fig = Figure()
    E = zeros(length(allN))
    for (fun,deriv,title,pos,rate) in data
        for (k,N) in enumerate(allN)
            h = 2π / N
            x = h * (1:N)
            column = [0; @. 0.5 * (-1)^(1:N-1) * cot((1:N-1) * h / 2)]
            D = toeplitz(column, column[[1; N:-1:2]])
            E[k] = norm(D * fun.(x) - deriv.(x), Inf)
        end
        ax = Axis(fig[pos[1], pos[2]],
            title=title, yscale=log10,
            xticks=0:10:Nmax, yticks=LogTicks(LinearTicks(4)),
        )
        scatterlines!(allN, E)
        thm = @. rate( 2π / allN )
        scatterlines!(allN, thm/thm[1]*E[1])
        limits!(0, Nmax, 1e-16, 1e3)
        ax.xlabel = (pos[1] == 2) ? "N" : ""
        ax.ylabel = (pos[2] == 1) ? "error" : ""
    end
    return fig
end

function prob4_7()
    L = 8                             # domain is [-L L], periodic
    N = 20:10:100 
    λ = zeros(20, length(N))
    for (j,N) in enumerate(N)
        h = 2π / N
        x = h * (1:N)
        x = @. L * (x - π) / π
        column = [-π^2 / 3h^2 - 1 / 6; @. -0.5 * (-1)^(1:N-1) / sin(h * (1:N-1) / 2)^2]
        D2 = (π / L)^2 * toeplitz(column)  # 2nd-order differentiation
        eigenvalues = sort(eigvals(-D2 + diagm(x .^ 4)))
        λ[:,j] = eigenvalues[1:20]
    end
    return λ
end
