module SpectralMethodsTrefethen

using MATLAB
using LinearAlgebra, SparseArrays, Dierckx, Polynomials, LaTeXStrings, SpecialFunctions, FFTW
using Plots, Printf

# Analogs for named functions in the book.
export cheb, chebfft, clencurt, gauss, chebinterp
include("bookfuns.jl")

# Stand-ins for native functions in MATLAB.
include("stand_ins.jl")

# See if MATLAB is available/working and can find the original files.
is_matlab_running = nothing
function look_for_matlab()
    try
        @info "Checking for MATLAB..."
        d = joinpath(@__DIR__,"..","mfiles")
        eval_string("mfiledir = cd('$d');")
        @mget mfiledir
        # At some point MATLAB started warning for 2D 'cubic' interp.
        eval_string("warning off MATLAB:griddedInterpolant:CubicUniformOnlyWarnId");
        @info "...success."
        global is_matlab_running = true
    catch
        @warn "MATLAB is not available. Continuing without MATLAB."
        global is_matlab_running = false
    end
end

## Create callable functions for each of the Julia scripts.

export p1, p2, p3, p4, p5, p6, p7, p8, p9, p10
export p11, p12, p13, p14, p15, p16, p17, p18, p19, p20
export p21, p22, p23, p24, p25, p26, p27, p28, p29, p30
export p31, p32, p33, p34, p35, p36, p37, p38, p39, p40
export p6u, p6anim, p19anim, p20anim, p23a, p27anim, p28b, p30b, p30c, p34anim, p35anim, p37anim
for (root, dirs, files) in walkdir(joinpath(@__DIR__,"scripts"))
    for file in filter(x->endswith(x,".jl"),files)
        basename = file[1:end-3];
        fundef = quote
            function $(Symbol(basename))(;julia=true,matlab=false,source=false)
                if julia
                    # Packages are needed in Main scope for include()
                    #@info join(["Running julia version of ",$basename,"..."])
                    if source 
                        for line in eachline(joinpath($root,$file))
                            println(line)
                        end
                        println("")
                    end
                    close("all")
                    pyplot()
                    default(legend=false,titlefontsize=10,xguidefontsize=8,xtickfontsize=7,yguidefontsize=8,ytickfontsize=7,zguidefontsize=8,ztickfontsize=6,grid=true)
                    include(joinpath($root,$file))
                end
                if matlab
                    if isnothing(SpectralMethodsTrefethen.is_matlab_running)
                        SpectralMethodsTrefethen.look_for_matlab()
                    end
                    if !SpectralMethodsTrefethen.is_matlab_running
                        @warn "MATLAB cannot be run."
                    else
                        #@info join(["Running MATLAB version of ",$basename,"..."])
                        if source 
                            for line in eachline(joinpath($root,"..","..","mfiles",$basename * ".m"))
                                println(line)
                            end
                            println("")
                        end
                        eval_string("close all")
                        eval_string($basename);
                     end
                end
                return plt
            end
        end
        eval(fundef);
    end
end

end
