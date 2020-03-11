"""
	baryinterp(u,x)
	baryinterp(u,x,w)
Create a polynomial interpolant by the barycentric formula for the function values in vector `u` at the node locations in vector `x`. If given, `w` is a vector of the barycentric weights; otherwise it is computed from the nodes. The return value is a callable function of the interpolation variable. 
"""
function baryinterp(u,x,w)
    n = length(u)
    t = zeros(n)
    hit = falses(n)
    return function(s)
        for i in 1:n 
            t[i] = w[i]/(s-x[i])
            hit[i] = isinf(t[i])
        end
        any(hit) ? u[findfirst(hit)] : dot(t,u)/sum(t)
    end
end 

function baryinterp(u,x)
    n = length(u)
    w = [ 1/prod(2(x[j]-x[k]) for k in 1:n if k !== j) for j in 1:n ]
    baryinterp(u,x,w)
end

"""
    chebinterp(v)
Create a callable interpolant for the vector `v` of values given at Chebyshev nodes in [-1,1].
"""
function chebinterp(u)
    n = length(u)-1
    wc = [ 0.5; (-1).^(1:n-1); 0.5*(-1)^n ]
    xc = [ cos(k*π/n) for k in 0:n ]
    baryinterp(u,xc,wc)
end

"""
    fourinterp(v)
Create a callable interpolant for the vector `v` of values given at equispaced (Fourier) nodes in (0,2π].
"""
function fourinterp(v)
    N = length(v)
    isodd(N) && @error "N must be even"
    SN = x -> sin(N*x/2) / (N*tan(x/2))  # δ interpolant 
    return function(x)
        t = [ SN(x-m*2π/N) for m in 1:N ]
        hit = isnan.(t)
        any(hit) ? v[findfirst(hit)] : dot(v,t)
    end
end

