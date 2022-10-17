## Analogs of functions from the book.
##
"""
    cheb(N)

Chebyshev differentiation matrix and grid.
"""
function cheb(N)
    N == 0 && return 0, 1
    x = [cos(π * k / N) for k in 0:N]
    function coeff(k)
        (k == 0) || (k == N) ? 2 : 1
    end
    # For off-diagonal entries:
    D = zeros(N + 1, N + 1)
    for i in 0:N
        for j in [0:i-1; i+1:N]
            D[i+1, j+1] = coeff(i) / coeff(j) * (-1)^(i + j) / (x[i+1] - x[j+1])
        end
        D[i+1, i+1] = -sum(D[i+1, :])
    end
    return D, x
end

"""
    chebfft(v)

Differentiate values given at Chebyshev points via the FFT.
"""
function chebfft(v)
    # Simple, not optimal.
    N = length(v) - 1
    N == 0 && return 0
    x = [cos(π * k / N) for k in 0:N]
    V = [v; v[N:-1:2]]
    U = real(fft(V))
    W = ifft(1im * [0:N-1; 0; 1-N:-1] .* U)
    isreal(v) && (W = real(W))
    w = zeros(eltype(v), N + 1)
    @. w[2:N] = -W[2:N] / sqrt(1 - x[2:N]^2)    # transform theta -> x
    w[1] = sum(i^2 * U[i+1] for i in 0:N-1) / N + 0.5 * N * U[N+1]
    s = sum((-1)^(i + 1) * i^2 * U[i+1] for i in 0:N-1)
    w[N+1] = s / N + 0.5 * (-1)^(N + 1) * N * U[N+1]
    return w
end

"""
    clencurt(N)

Nodes and weights for Clenshaw-Curtis quadrature
"""
function clencurt(N)
    θ = [π * i / N for i in 0:N]
    x = cos.(θ)
    w = zeros(N + 1)
    v = ones(N-1)
    for k = 1:(N-1)÷2
        v -= 2cos.(2 * k * θ[2:N]) / (4 * k^2 - 1)
    end
    if iseven(N)
        w[1] = w[N+1] = 1 / (N^2 - 1)
        v -= cos.(N * θ[2:N]) / (N^2 - 1)
    else
        w[1] = w[N+1] = 1 / N^2
    end
    w[2:N] .= 2v / N
    return x, w
end

"""
    gauss(N)

Nodes and weights for Gauss quadrature
"""
function gauss(N)
    β = [0.5 / sqrt(1 - 1 / (2i)^2) for i in 1:N-1]
    T = diagm(-1 => β, 1 => β)
    x, V = eigen(T)
    i = sortperm(x)
    w = 2 * V[1, i] .^ 2
    return x[i], w
end

##
## Stand-ins for native functions in MATLAB.
##

import Base.view
export toeplitz, view
"""
    toeplitz(col[,row])

Construct Toeplitz matrix from first column and first row. If the row is not
given, the result is symmetric.
"""
function toeplitz(col, row=col)
    m, n = length(col), length(row)
    col[1] == row[1] || warn("Column wins conflict on the diagonal.")
    x = [row[end:-1:2]; col]
    return [x[i-j+n] for i in 1:m, j in 1:n]
end

"""
    view(az,el)

Sets the 3D viewing orientation azimuth and elevation (in degrees).
"""
function view(az::Real, el::Real)
    gca()[:view_init](el, az - 90)
end
