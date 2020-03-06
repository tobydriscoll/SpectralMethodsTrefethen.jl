module SpectralMethodsTrefethen

using MATLAB
using LinearAlgebra, SparseArrays, Dierckx, Polynomials, LaTeXStrings, SpecialFunctions, FFTW
using Plots, Printf
#import Contour

# Analogs for named functions in the book.
export cheb, chebfft, clencurt, gauss

# # Required by the scripts, so must be in parent scope
# eval(parentmodule(@__MODULE__),:(using PyPlot, Dierckx, Polynomials, LaTeXStrings, SpecialFunctions))

## Analogs of functions from the book.
include("bookfuns.jl")

## Stand-ins for native functions in MATLAB.
include("stand_ins.jl")

# Indicate that MATLAB has not yet been looked for.
is_matlab_running = nothing

# See if MATLAB is available/working and can find the original files.
function look_for_matlab()
    try
        @info "Checking for MATLAB..."
        d = joinpath(@__DIR__,"..","mfiles")
        eval_string("mfiledir = cd('$d');")
        @mget mfiledir
        # At some point MATLAB started warning for 2D 'cubic' interp.
        eval_string("warning off MATLAB:griddedInterpolant:CubicUniformOnlyWarnId");
        @info "...success."
        true
    catch
        @warn "MATLAB is not available. Continuing without MATLAB."
        false
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
                println(join(["Running script ",$basename,"..."]))
                t = []
                if julia
                    # Packages are needed in Main scope for include()
                    @info "Julia version..."
                    close("all")
                    pyplot()
                    default(legend=false,titlefontsize=10,xguidefontsize=8,xtickfontsize=7,yguidefontsize=8,ytickfontsize=7,zguidefontsize=8,ztickfontsize=6,grid=true)
                    tt = @timed include(joinpath($root,$file))
                    #println(" finished in $(tt[2]) seconds")
                    push!(t,tt[2])
                end
                if matlab
                    if isnothing(is_matlab_running)
                        is_matlab_running = SpectralMethodsTrefethen.look_for_matlab()
                    end
                    if !is_matlab_running
                        @warn "MATLAB cannot be run."
                        return t
                    end
                    @info "MATLAB version..."
                    eval_string("close all")
                    eval_string("pwd")
                    tt = @timed eval_string($basename)
                    #println(" finished in $(tt[2]) seconds")
                    push!(t,tt[2])
                end
                if source
                    if julia 
                        for line in eachline(joinpath($root,$file))
                            println(line)
                        end
                    end
                    if matlab 
                        for line in eachline(joinpath($root,"..","matlab",$basename * ".m"))
                            println(line)
                        end
                    end
                end
                return plt
            end
        end
        eval(fundef);
    end
end

end
