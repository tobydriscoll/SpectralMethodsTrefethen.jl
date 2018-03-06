module SpectralMethodsTrefethen

# Analogs for named functions in the book.
export cheb, chebfft, clencurt, gauss

# Required by the scripts.
using PyPlot, Interpolations, Polynomials, LaTeXStrings, SpecialFunctions

# See if MATLAB is available/working.
is_matlab_running = try
    using MATLAB
    true
catch exc
    warn("MATLAB is not available. Continuing without MATLAB.");
    false
end

# Now see if it can find our files.
if is_matlab_running
    is_matlab_running = try
        local d = joinpath(Pkg.dir("SpectralMethodsTrefethen"),"matlab");
        eval_string("mfiledir = cd('$d');");
        @mget mfiledir;
        true
    catch
        warn("MATLAB cannot find package files. Continuing without MATLAB.");
        false
    end
end

## Analogs of functions from the book.

"""
    cheb(N)

Chebyshev differentiation matrix and grid.
"""
function cheb(N)
    N==0 && return 0,1;
    x = [ cos(pi*k/N) for k=0:N ];
    c = [2;ones(N-1);2] .* (-1).^(0:N);
    dX = x .- x';
    D  = (c*(1./c)') ./ (dX+eye(N+1));      # off-diagonal entries
    D  = D - diagm(squeeze(sum(D,2),2));    # diagonal entries
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
    V = [v; flipdim(v[2:N],1)];              # transform x -> theta
    U = real(fft(V));
    W = real(ifft(1im*[ii;0;1-N:-1].*U));
    w = zeros(N+1);
    w[2:N] = -W[2:N]./sqrt.(1-x[2:N].^2);    # transform theta -> x
    w[1] = sum(ii.^2.*U[ii+1])/N + .5*N*U[N+1];
    w[N+1] = sum((-1).^(ii+1).*ii.^2.*U[ii+1])/N + .5*(-1)^(N+1)*N*U[N+1];
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
    T = diagm(β,1) + diagm(β,-1);
    (x,V) = eig(T);
    i = sortperm(x);
    w = 2*V[1,i].^2;
    return x[i],w
end

## Stand-in for the native toeplitz function in MATLAB.
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

## Create callable functions for each of the Julia scripts.

export p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
export p11, p12, p13, p14, p15, p16, p17, p18, p19, p20
export p21, p22, p23, p24, p25, p26, p27, p28, p29, p30
export p31, p32, p33, p34, p35, p36, p37, p38, p39, p40
export p6u, p23a, p24fine, p28b, p30b, p30c
for (root, dirs, files) in walkdir(joinpath(Pkg.dir("SpectralMethodsTrefethen"),"src","scripts"))
    for file in files
        ismatch(r"\.jl$",file) || continue
        basename = file[1:end-3];
        fundef = quote
            function $(Symbol(basename))(;julia=true,matlab=$is_matlab_running,source=false)
                println(join(["Running script ",$basename,"..."]));
                if julia
                    println("Julia version:");
                    tic(); include(joinpath($root,$file)); t=toc();
                end
                if matlab
                    println("MATLAB version:");
                    tic(); eval_string($basename); t=[t;toc()];
                end
                if source
                    julia ? edit(joinpath($root,$file)) : nothing;
                    matlab ? edit(joinpath($root,"..","..","matlab",$basename * ".m")) : nothing;
                end
                return t

                function thescript()

            end
        end
        eval(fundef);
    end
end

end
