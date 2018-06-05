module SpectralMethodsTrefethen

# Analogs for named functions in the book.
export cheb, chebfft, clencurt, gauss

# Required by the scripts, so must be in parent scope
eval(parentmodule(@__MODULE__),:(using PyPlot, Dierckx, Polynomials, LaTeXStrings, SpecialFunctions))

## Analogs of functions from the book.
include("bookfuns.jl")

## Stand-ins for native functions in MATLAB.
include("stand_ins.jl")


# See if MATLAB is available/working and can find the original files.
is_matlab_running = try
    local d = joinpath(@__DIR__,"mfiles");
    eval_string("mfiledir = cd('$d');");
    @mget mfiledir;
    # At some point MATLAB started warning for 2D 'cubic' interp.
    eval_string("warning off MATLAB:griddedInterpolant:CubicUniformOnlyWarnId");
    true
catch
    warn("MATLAB is not available. Continuing without MATLAB.");
    false
end

## Create callable functions for each of the Julia scripts.

export p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
export p11, p12, p13, p14, p15, p16, p17, p18, p19, p20
export p21, p22, p23, p24, p25, p26, p27, p28, p29, p30
export p31, p32, p33, p34, p35, p36, p37, p38, p39, p40
export p6u, p23a, p24fine, p28b, p30b, p30c
for (root, dirs, files) in walkdir(joinpath(@__DIR__,"src","scripts"))
    for file in filter(x->endswith(x,".jl"),files)
        basename = file[1:end-3];
        fundef = quote
            function $(Symbol(basename))(;julia=true,matlab=$is_matlab_running,source=false)
                println(join(["Running script ",$basename,"..."]));
                if julia
                    # Packages are needed in Main scope for include()
                    println("Julia version:");
                    close("all")
                    tic(); include(joinpath($root,$file)); t=toc();
                end
                if matlab
                    println("MATLAB version:");
                    eval_string("close all");
                    tic(); eval_string($basename); t=[t;toc()];
                end
                if source
                    julia ? edit(joinpath($root,$file)) : nothing;
                    matlab ? edit(joinpath($root,"..","..","matlab",$basename * ".m")) : nothing;
                end
                return t

            end
        end
        eval(fundef);
    end
end

end
