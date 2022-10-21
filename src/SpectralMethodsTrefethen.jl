module SpectralMethodsTrefethen

# Analogs for named functions in the book.
export cheb, chebfft, clencurt, gauss

using LinearAlgebra, SparseArrays
using PyPlot
# using CairoMakie,GLMakie
using Dierckx
using FFTW
using Polynomials
using LaTeXStrings
using SpecialFunctions
using PrettyTables
import Base:view

export p1, p2, p3, p3g, p4, p5, p5r, p6, p7, p8, p9, p10
export p11, p12, p13, p14, p15, p16, p17, p18, p19, p20
export p21, p22, p23, p24, p25, p26, p27, p28, p29, p30
export p31, p32, p33, p34, p35, p36, p37, p38, p39, p40
export p6u, p23a, p24fine, p28b, p30b, p30c

export cheb, chebfft, clencurt, gauss, polyinterp, toeplitz, view

# pygui(true)

include("utils.jl")
include("codes.jl")

# set_theme!(SMT_theme)
# CairoMakie.activate!()

end
