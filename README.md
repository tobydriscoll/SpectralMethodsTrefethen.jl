# SpectralMethodsTrefethen

[![Build Status](https://travis-ci.org/tobydriscoll/SpectralMethodsTrefethen.jl.svg?branch=master)](https://travis-ci.org/tobydriscoll/SpectralMethodsTrefethen.jl)

[![Coverage Status](https://coveralls.io/repos/tobydriscoll/SpectralMethodsTrefethen.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tobydriscoll/SpectralMethodsTrefethen.jl?branch=master)

[![codecov.io](http://codecov.io/github/tobydriscoll/SpectralMethodsTrefethen.jl/coverage.svg?branch=master)](http://codecov.io/github/tobydriscoll/SpectralMethodsTrefethen.jl?branch=master)


This is a Julia 1.x rewrite of the codes in Trefethen's *Spectral Methods in MATLAB.* It's in alpha and the codes are not at all thoroughly checked yet, much less made beautiful.

The graphics is done by `PyPlot`, which is the most MATLAB-like interface (but not the one most used in Julia).

# Installation

I recommend that you create a separate Julia environment to work with this package. Change to a new directory for it and enter

```julia
import Pkg; Pkg.activate(".")
```

to create the environment. Then

```julia
Pkg.add(url="https://github.com/tobydriscoll/SpectralMethodsTrefethen.jl")
Pkg.add("PyPlot")
```

# Usage

To get the package loaded, activate the dedicated environment, then enter

```julia
using SpectralMethodsTrefethen,PyPlot
pygui(true)
```

Then each of the book's program scripts can be run as functions, e.g. `p1()`, `p2()`, etc. Some of the functions have additional versions with a suffix like `b` or `u`.

The book functions `cheb`, `chebfft`, `clencurt` and `gauss` are also available.

# Finding sources

As with any Julia function, you can use the `@edit` macro, like in

```julia
@edit p1()
```
