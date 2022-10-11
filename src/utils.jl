## Analogs of functions from the book.
##
"""
    cheb(N)

Chebyshev differentiation matrix and grid.
"""
function cheb(N)
    N==0 && return 0,1;
    x = [ cos(pi*k/N) for k=0:N ];
    c = [2;ones(N-1);2] .* (-1).^(0:N);
    dX = x .- x';
    D  = (c*(1.0./c)') ./ (dX+I(N+1));      # off-diagonal entries
    D  = D - diagm(vec(sum(D,dims=2)));    # diagonal entries
    return D,x
end


"""
    chebfft(v)

Differentiate values given at Chebyshev points via the FFT.
"""
function chebfft(v)
    # Simple, not optimal. If v is complex, delete "real" commands.
    N = length(v)-1;
    N==0 && return 0;
    x = [ cos(pi*k/N) for k=0:N ];
    ii = collect(0:N-1);
    V = [v; reverse(v[2:N],dims=1)];              # transform x -> theta
    U = real(fft(V));
    W = real(ifft(1im*[ii;0;1-N:-1].*U));
    w = zeros(N+1);
    @. w[2:N] = -W[2:N]/sqrt(1-x[2:N]^2);    # transform theta -> x
    w[1] = sum(ii.^2 .* U[ii.+1])/N + .5*N*U[N+1];
    w[N+1] = sum((-1).^(ii.+1).*ii.^2 .* U[ii.+1])/N + .5*(-1)^(N+1)*N*U[N+1];
    return w
end

"""
    clencurt(N)

Nodes and weights for Clenshaw-Curtis quadrature
"""
function clencurt(N)
    θ = [ pi*i/N for i=0:N ];
    x = cos.(θ);
    w = zeros(N+1);
    ii = 2:N;
    v = ones(N-1);
    if mod(N,2)==0
        w[1] = w[N+1] = 1/(N^2-1);
        for k = 1:N/2-1
            v = v - 2*cos.(2*k*θ[ii]) / (4*k^2-1);
        end
        v = v - cos.(N*θ[ii]) / (N^2-1);
    else
        w[1] = w[N+1] = 1/N^2;
        for k = 1:(N-1)/2
            v = v - 2*cos.(2*k*θ[ii]) / (4*k^2-1);
        end
    end
    w[ii] = 2*v/N;
    return x,w
end

"""
    gauss(N)

Nodes and weights for Gauss quadrature
"""
function gauss(N)
    β = [ .5/sqrt(1-1/(2*i)^2) for i = 1:N-1 ];
    T = diagm(1=>β) + diagm(-1=>β);
    x,V = eigen(T);
    i = sortperm(x);
    w = 2*V[1,i].^2;
    return x[i],w
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
function toeplitz(col,row=col)
    m,n = length(col),length(row);
    col[1]==row[1] || warn("Column wins conflict on the diagonal.");
    x = [ row[end:-1:2]; col ];
    return [ x[i-j+n] for i=1:m, j=1:n ]
end

"""
    view(az,el)

Sets the 3D viewing orientation azimuth and elevation (in degrees).
"""
function view(az::Real,el::Real)
    gca()[:view_init](el,az-90);
end
