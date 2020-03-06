## Analogs of functions from the book.
"""
    cheb(N)

Chebyshev differentiation matrix and grid.
"""
function cheb(N)
    N==0 && return zeros(1,1),[1]
    x = [ cos(π*k/N) for k=0:N ]
    c = [2;ones(N-1);2] .* (-1).^(0:N)
    dX = x .- x'
    D = (c*(1 ./ c)') ./ (dX+I);      # off-diagonal entries
    D -= Diagonal(dropdims(sum(D,dims=2),dims=2))   # diagonal entries
    return D,x
end


"""
    chebfft(v)

Differentiate values given at Chebyshev points via the FFT.
"""
function chebfft(v)
    # Simple, not optimal. If v is complex, delete "real" commands.
    N = length(v)-1
    N==0 && return 0
    x = [ cos(pi*k/N) for k=0:N ]
    ii = 0:N-1
    V = [v; reverse(v[2:N],dims=1)]              # transform x -> theta
    U = real(fft(V))
    W = real(ifft(1im*[ii;0;1-N:-1].*U))
    w = zeros(N+1)
    w[2:N] = -W[2:N]./sqrt.(1 .- x[2:N].^2)      # transform theta -> x
    w[1] = sum(i^2*U[i+1] for i in ii)/N + .5*N*U[N+1]
    w[N+1] = sum((-1)^(i+1)*i^2*U[i+1] for i in ii)/N + .5*(-1)^(N+1)*N*U[N+1]
    return w
end

"""
    clencurt(N)

Nodes and weights for Clenshaw-Curtis quadrature
"""
function clencurt(N)
    θ = [ pi*i/N for i=0:N ]
    x = cos.(θ)
    w = zeros(N+1)
    ii = 2:N
    v = ones(N-1)
    if iseven(N)
        w[1] = w[N+1] = 1/(N^2-1)
        for k = 1:N/2-1
            v -= 2*cos.(2*k*θ[ii]) / (4*k^2-1)
        end
        v -= cos.(N*θ[ii]) / (N^2-1)
    else
        w[1] = w[N+1] = 1/N^2
        for k = 1:(N-1)/2
            v -= 2*cos.(2*k*θ[ii]) / (4*k^2-1)
        end
    end
    w[ii] = 2v/N
    return x,w
end

"""
    gauss(N)

Nodes and weights for Gauss quadrature
"""
function gauss(N)
    β = [ .5/sqrt(1-1/(2*i)^2) for i = 1:N-1 ]
    T = SymTridiagonal(zeros(N),β)
    x,V = eigen(T)
    return x,2*V[1,:].^2
end


"""
    chebinterp(vals)

Create an interpolant for values given at Chebyshev nodes
"""
function chebinterp(u,x,w)
    n = length(u)
    t = zeros(n)
    hit = falses(n)
    return function(s)
        for i in 1:n 
            @inbounds t[i] = w[i]/(s-x[i])
            @inbounds hit[i] = isinf(t[i])
        end
        any(hit) ? u[findfirst(hit)] : dot(t,u)/sum(t)
    end
end 

function chebinterp(u,x)
    n = length(u)
    w = [ 1/prod(2(x[j]-x[k]) for k in 1:n if k !== j) for j in 1:n ]
    chebinterp(u,x,w)
end

function chebinterp(u)
    n = length(u)-1
    wc = [ 0.5; (-1).^(1:n-1); 0.5*(-1)^n ]
    xc = [ cos(k*π/n) for k in 0:n ]
    chebinterp(u,xc,wc)
end

function chebinterp_slow(u)
    n = length(u)
    wc = k -> 1 < k < n ? (-1)^(n-k) : (-1)^(n-k)/2
    xc = k -> sin(π*(2k-n-1)/(2n-2))
    return function(x)
        t = [ wc(i)/(x-xc(i)) for i in 1:n ]
        hit = isinf.(t)
        any(hit) ? u[findfirst(hit)] : dot(t,u)/sum(t)
    end
end 
