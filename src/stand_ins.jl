## Stand-ins for native functions in MATLAB.

using PyPlot: gca  # needed for view() below
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
