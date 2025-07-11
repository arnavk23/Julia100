# [![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Farnavk23%2FJulia250Exercises&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors+%28Daily+%2F+Total%29&edge_flat=false)](https://github.com/arnavk23/Julia250Exercises)
#
# # 100 Julia Exercises with Solutions
#
# An extended set of Julia exercises for comprehensive learning. Inspired by [Julia100Exercises](https://github.com/RoyiAvital/Julia100Exercises).
#
# **Author**: Arnav Kapoor ([@arnavk23](https://github.com/arnavk23))
#
# In order to generate this file:
# 1. Clone the repository (Or download).
# 2. Activate and instantiate the project.
# 3. Run:
# ```Julia
# using Literate;
# Literate.markdown("Julia100Exercises.jl", name = "README", execute = true, flavor = Literate.CommonMarkFlavor());
# ```
#
# **Remark**: Tested with Julia `1.10`.  
#
# using Literate;  # Commented out - only needed for documentation generation
using LinearAlgebra;
using Statistics;
using Dates;
using DelimitedFiles;
# using UnicodePlots;  # Commented out - optional for plotting
using Random;
# using Tullio;  # Commented out - advanced tensor operations package
# using StaticKernels;  # Commented out - specialized kernels package

# ## Question 001
# Import the `LinearAlgebra` package under the name `LA`. (Easy)

import LinearAlgebra as LA;

# ## Question 002
# Print the version of Julia. (Easy)

println(VERSION);

# ## Question 003
# Create a non initialized vector of size 10 of `Float64`. (Easy)

vA = Vector{Float64}(undef, 10)

# Which is equivalent to (though the first form is preferred in Julia 1.10+):

vA = Array{Float64, 1}(undef, 10)

# Modern Julia 1.10+ also supports:
vA = Vector{Float64}(undef, 10)

# ## Question 004
# Find the memory size of any array. (Easy)

sizeof(vA)

# ## Question 005
# Show the documentation of the `+` (Add) method. (Easy)

@doc +

# ## Question 006
# Create a vector of zeros of size 10 but the fifth value which is 1. (Easy)

vA = zeros(10);
vA[5] = 1.0;
vA

# ## Question 007
# Create a vector with values ranging from 7 to 12. (Easy)

vA = 7:12

# The above creates an efficient range type. To explicitly create a vector:

vA = collect(7:12)

# In Julia 1.10+, you can also use:
vA = [7:12...]

# ## Question 008
# Reverse a vector (first element becomes last). (Easy)

vA = collect(1:3);
vB = vA[end:-1:1];
vB

# Alternative 001:

vB = reverse(vA);

# Alternative 002 (In place):

reverse!(vA);
vA

# ## Question 009
# Create a `3x3` matrix with values ranging from 0 to 8. (Easy)

mA = reshape(0:8, 3, 3)

# Another way:

mA = Matrix{Float64}(undef, 3, 3);
mA[:] = 0:8;
mA

# Modern Julia 1.10+ alternative:
mA = collect(reshape(0:8, 3, 3))

# ## Question 010
# Find indices of non zero elements from `[1, 2, 0, 0, 4, 0]`. (Easy)

findall(!iszero, [1, 2, 0, 0, 4, 0])

# ## Question 011
# Create a 3x3 identity matrix. (Easy)

mA = I(3)

# An alternative method (explicit matrix):

mA = Matrix(I, 3, 3) # For Float64: Matrix{Float64}(I, 3, 3)

# Julia 1.10+ also supports:
mA = diagm(ones(3))

# ## Question 012
# Create a `2x2x2` array with random values. (Easy)

mA = randn(2, 2, 2)

# ## Question 013
# Create a `5x5` array with random values and find the minimum and maximum values. (Easy)

mA = rand(5, 5);
minVal = minimum(mA)
#+
maxVal = maximum(mA)

# Using `extrema()` one could get both values at once:

minVal, maxVal = extrema(mA);

# ## Question 014
# Create a random vector of size 30 and find the mean value. (Easy)

meanVal = mean(randn(30))

# ## Question 015
# Create a 2d array with 1 on the border and 0 inside. (Easy)

mA = zeros(4, 4);
mA[:, [1, end]] .= 1;
mA[[1, end], :] .= 1;
mA

# An alternative way (Different dimensions):

mA = ones(4, 5);
mA[2:(end - 1), 2:(end - 1)] .= 0;

# Using one line code:

mA = zeros(4, 5);
mA[[LinearIndices(mA)[cartIdx] for cartIdx in CartesianIndices(mA) if (any(cartIdx.I .== 1) || cartIdx.I[1] == size(mA, 1) || cartIdx.I[2] == size(mA, 2))]] .= 1;

# By [Tomer Arnon](https://github.com/tomerarnon):

numRows = 5;
numCols = 4;
mA = Int[ii ∈ (1, numRows) || jj ∈ (1, numCols) for ii in 1:numRows, jj in 1:numCols];

# ## Question 016
# Add a border of zeros around the array. (Easy)

mB = zeros(size(mA) .+ 2);
mB[2:(end - 1), 2:(end - 1)] = mA;
mB

# ## Question 017
# Evaluate the following expressions. (Easy)

0 * NaN
#+
NaN == NaN
#+
Inf > NaN
#+
NaN - NaN
#+
NaN in [NaN]
#+
0.3 == 3 * 0.1

# ## Question 018
# Create a `5x5` matrix with values `[1, 2, 3, 4]` just below the diagonal. (Easy)

mA = diagm(5, 5, -1 => 1:4)

# Alternative in Julia 1.10+:
mA = diagm(-1 => 1:4, size=(5,5))

# ## Question 019
# Create a `8x8` matrix and fill it with a checkerboard pattern. (Easy)

mA = zeros(8, 8);
mA[2:2:end, 1:2:end] .= 1;
mA[1:2:end, 2:2:end] .= 1;
mA

# By Tomer Arnon (https://github.com/tomerarnon):

mA = Int[isodd(ii + jj) for ii in 1:8, jj in 1:8];

# ## Question 020
# Convert the linear index 100 to a _Cartesian Index_ of a size `(6,7,8)`. (Easy)

mA = rand(6, 7, 8);
cartIdx = CartesianIndices(mA)[100]; # See https://discourse.julialang.org/t/14666
mA[cartIdx] == mA[100]

# Alternative using Julia 1.10+ syntax:
cartIdx = CartesianIndices((6, 7, 8))[100]

# ## Question 021
# Create a checkerboard `8x8` matrix using the `repeat()` function. (Easy)

mA = repeat([0 1; 1 0], 4, 4)

# ## Question 022
# Normalize a `4x4` random matrix. (Easy)

mA = rand(4, 4);
mA .= (mA .- mean(mA)) ./ std(mA) # Note: @. macro would cause error with std() and mean()

# Alternative using the @. macro properly:
mA = rand(4, 4);
μ = mean(mA);
σ = std(mA);
@. mA = (mA - μ) / σ

# ## Question 023
# Create a custom type that describes a color as four unsigned bytes (`RGBA`). (Easy)

struct sColor
    R::UInt8;
    G::UInt8;
    B::UInt8;
    A::UInt8;
end

sMyColor = sColor(rand(UInt8, 4)...)

# ## Question 024
# Multiply a `2x4` matrix by a `4x3` matrix. (Easy)

mA = rand(2, 4) * randn(4, 3)

# ## Question 025
# Given a 1D array, negate all elements which are between 3 and 8, in place. (Easy)

vA = rand(1:10, 8);
map!(x -> ((x > 3) && (x < 8)) ? -x : x, vA, vA)

# Julia allows math-like notation as well (see Q0027):

vA = rand(1:10, 8);
map!(x -> 3 < x < 8 ? -x : x, vA, vA)

# Using logical indices (modern Julia 1.10+ preferred):

vA = rand(1:10, 8);
vA[3 .< vA .< 8] .*= -1;

# ## Question 026
# Sum the array `1:4` with initial value of -10. (Easy)

sum(1:4, init = -10)

# ## Question 027
# Consider an integer vector `vZ` validate the following expressions. (Easy)
# ```julia
# vZ .^ vZ
# 2 << vZ >> 2
# vZ <- vZ
# 1im * vZ
# vZ / 1 / 1
# vZ < Z > Z
# ```

vZ = rand(1:10, 3);
#+
vZ .^ vZ
#+
try
    2 << vZ >> 2
catch e
    println(e)
end
#+
vZ <- vZ
#+
1im * vZ
#+
vZ / 1 / 1
#+
vZ < vZ > vZ

# ## Question 028
# Evaluate the following expressions. (Easy)

[0] ./ [0]
#+
try
    [0] .÷ [0]
catch e
    println(e)
end
#+
try
    convert(Float64, convert(Int, NaN))  # Specify Float64 explicitly in Julia 1.10+
catch e
    println(e)
end

# ## Question 029
# Round away from zero a float array. (Easy)

vA = randn(10);
map(x -> x > 0 ? ceil(x) : floor(x), vA)

# Alternative using sign function (Julia 1.10+):
vA = randn(10);
sign.(vA) .* ceil.(abs.(vA))

# ## Question 030
# Find common values between two arrays. (Easy)

vA = rand(1:10, 6);
vB = rand(1:10, 6);

vA[findall(in(vB), vA)]

# Alternative using intersect (Julia 1.10+):
intersect(vA, vB)

# Or using Set operations:
collect(Set(vA) ∩ Set(vB))

# ## Question 031
# Suppress Julia's warnings. (Easy)

# One could use [Suppressor.jl](https://github.com/JuliaIO/Suppressor.jl).
# Or in Julia 1.10+, use logging control:
using Logging
with_logger(NullLogger()) do
    # Code that would generate warnings
end

# ## Question 032
# Compare `sqrt(-1)` and `sqrt(-1 + 0im)`. (Easy)

try
    sqrt(-1)
catch e
    println(e)
end
#+
sqrt(-1 + 0im)

# ## Question 033
# Display yesterday, today and tomorrow's date. (Easy)

println("Yesterday: $(today() - Day(1))");
println("Today: $(today())");
println("Tomorrow: $(today() + Day(1))");

# ## Question 034
# Display all the dates corresponding to the month of July 2016. (Medium)

collect(Date(2016,7,1):Day(1):Date(2016,7,31))

# ## Question 035
# Compute `((mA + mB) * (-mA / 2))` in place. (Medium)

mA = rand(2, 2);
mB = rand(2, 2);
mA .= ((mA .+ mB) .* (.-mA ./ 2))

# Using the dot macro:

@. mA = ((mA + mB) * (-mA / 2));

# ## Question 036
# Extract the integer part of a random array of positive numbers using 4 different methods. (Medium)

mA = 5 * rand(3, 3);

# Option 1:
floor.(mA)

# Option 2:
round.(mA .- 0.5) # Generates -0.0 for numbers smaller than 0.5

# Option 3:
mA .÷ 1

# Option 4:
mA .- rem.(mA, 1)

# Option 5 (Julia 1.10+ using trunc):
trunc.(mA)

# ## Question 037
# Create a `5x5` matrix with row values ranging from 0 to 4. (Medium)

mA = repeat(reshape(0:4, 1, 5), 5, 1)

# One could also generate row-like range using transpose:
mA = repeat((0:4)', 5, 1);

# Julia 1.10+ alternative using broadcasting:
mA = zeros(Int, 5, 5) .+ (0:4)'

# ## Question 038
# Generate an array using a generator of 10 numbers. (Easy)

vA = collect(x for x in 1:10)

# In Julia, the result can be achieved directly using Array Comprehension:

vA = [x for x in 1:10];

# Julia 1.10+ simplification:
vA = collect(1:10)

# ## Question 039
# Create a vector of size 10 with values ranging from 0 to 1, both excluded. (Medium)

vA = LinRange(0, 1, 12)[2:(end - 1)]

# ## Question 040
# Create a random vector of size 10 and sort it. (Medium)

vA = rand(1:10, 10);
sort(vA) # Use `sort!()` for in-place sorting

# Julia 1.10+ also supports:
vA_sorted = sort!(copy(vA)) # Explicit copy for safety

# ## Question 041
# Implement the `sum()` function manually. (Medium)

vA = rand(100);

function MySum(vA::Vector{T}) where {T <: Number}
    sumVal = vA[1];
    for ii in 2:length(vA)
        sumVal += vA[ii];
    end
    return sumVal;
end

MySum(vA)

# Julia 1.10+ alternative using reduce:
function MySum2(vA::Vector{T}) where {T <: Number}
    reduce(+, vA; init=zero(T))
end

# ## Question 042
# Check for equality of 2 arrays. (Medium)

vA = rand(10);
vB = rand(10);

all(vA .== vB)

# ## Question 043
# Make an array immutable (Read only). (Medium)

# In Julia 1.10+, you can use:
# 1. Const arrays for compile-time immutability
const vImmutable = [1, 2, 3, 4, 5]

# 2. Or use ReadOnlyArrays.jl package for runtime immutability
# using ReadOnlyArrays
# vReadOnly = ReadOnlyArray([1, 2, 3, 4, 5])

# ## Question 044
# Consider a random `10x2` matrix representing cartesian coordinates, convert them to polar coordinates. (Medium)

mA = rand(10, 2);

ConvToPolar = vX -> [hypot(vX[1], vX[2]), atan(vX[2], vX[1])]

mB = [ConvToPolar(vX) for vX in eachrow(mA)]

# In order to have the same output size:

mC = reduce(hcat, mB)';

# ## Question 045
# Create random vector of size 10 and replace the maximum value by 0. (Medium)

vA = randn(10);

# In case of a single maximum or all different values:
vA[argmax(vA)] = 0;
vA

# General solution:

maxVal = maximum(vA);
vA .= (valA == maxVal ? 0 : valA for valA in vA); # Non-allocating generator by using `.=`

# Julia 1.10+ alternative:
vA = randn(10);
vA[vA .== maximum(vA)] .= 0;

# ## Question 046
# Create a grid of `x` and `y` coordinates covering the `[0, 1] x [0, 1]` area. (Medium)

numGridPts = 5;
vX = LinRange(0, 1, numGridPts);
vY = LinRange(0, 1, numGridPts);
MeshGrid = (vX, vY) -> ([x for _ in vY, x in vX], [y for y in vY, _ in vX]);

mX, mY = MeshGrid(vX, vY); # See https://discourse.julialang.org/t/48679
@show mX
#+
@show mY

# By [Tomer Arnon](https://github.com/tomerarnon):

mXY = [(ii, jj) for ii in 0:0.25:1, jj in 0:0.25:1]; # Also `tuple.(0:0.25:1, (0:0.25:1)')`

# Julia 1.10+ using Iterators.product:
using Base.Iterators
mXY_modern = collect(product(0:0.25:1, 0:0.25:1))

# ## Question 047
# Given two vectors, `vX` and `vY`, construct the Cauchy matrix `mC`: `(Cij = 1 / (xi - yj))`. (Medium)

vX = rand(5);
vY = rand(5);

mC = 1 ./ (vX .- vY')

# ## Question 048
# Print the minimum and maximum representable value for each Julia scalar type. (Medium)

vT = [UInt8 UInt16 UInt32 UInt64 Int8 Int16 Int32 Int64 Float16 Float32 Float64]

for juliaType in vT
    println(typemin(juliaType));
    println(typemax(juliaType));
end

# ## Question 049
# Print all the values of an array. (Medium)

mA = rand(3, 3);
print(mA);

# ## Question 050
# Find the closest value to a given scalar in a vector. (Medium)

inputVal = 0.5;
vA = rand(10);

vA[argmin(abs.(vA .- inputVal))]

# By [Tomer Arnon](https://github.com/tomerarnon):

function ClosestValue(vA::Vector{T}, inputVal::T) where {T <: Number}
    return vA[argmin(y -> abs(y - inputVal), vA)];  # Fixed: return the value, not index
end

ClosestValue(vA, inputVal)

# Julia 1.10+ using findmin:
_, idx = findmin(abs.(vA .- inputVal))
closest_value = vA[idx]

# ## Question 051
# Create a structured array representing a position `(x, y)` and a color `(r, g, b)`. (Medium)

struct sPosColor
    x::Int
    y::Int
    R::UInt8;
    G::UInt8;
    B::UInt8;
    A::UInt8;
end

numPixels   = 10;
maxVal      = typemax(UInt32);
vMyColor    = [sPosColor(rand(1:maxVal, 2)..., rand(UInt8, 4)...) for _ in 1:numPixels];

# ## Question 052
# Consider a random vector with shape `(5, 2)` representing coordinates, find the distances matrix `mD`: $ {D}_{i, j} = {\left\| {x}_{i} - {x}_{j} \right\|}_{2} $. (Medium)

mX = rand(5, 2);
vSumSqr = sum(vX -> vX .^ 2, mX, dims = 2);
mD = vSumSqr .+ vSumSqr' - 2 * (mX * mX');
mD # Apply `sqrt.()` for the actual norm

# Julia 1.10+ alternative using broadcasting:
mD_alt = sqrt.(sum((mX .- permutedims(mX, (2, 1))).^2, dims=2))

# ## Question 053
# Convert a float (32 bits) array into an integer (32 bits) in place. (Medium)

vA = 9999 .* rand(Float32, 5);
vB = reinterpret(Int32, vA); # Creates a view
@. vB = trunc(Int32, vA) # Updates the bytes in the view (In-place for `vA`)

# Julia 1.10+ alternative approach:
vA_copy = copy(vA);
vA_int = trunc.(Int32, vA_copy)

# The original approach is equivalent to:
# ```julia
# for ii in eachindex(vB)
#     vB[ii] = trunc(Int32, vA[ii]);
# end
# ```

# ## Question 054
# Read the following file (`Q0054.txt`). (Medium)
# ```
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ```

mA = readdlm("Q0054.txt", ',')

# Julia 1.10+ alternative using CSV.jl (if available):
# using CSV, DataFrames
# df = CSV.File("Q0054.txt", header=false) |> DataFrame

# ## Question 055
# Enumerate array in a loop. (Medium)

mA = rand(3, 3);

for (elmIdx, elmVal) in enumerate(mA) # See https://discourse.julialang.org/t/48877
    println("Index: $elmIdx, Value: $elmVal");
end

# Julia 1.10+ alternative using pairs():
for (idx, val) in pairs(mA)
    println("CartesianIndex: $idx, Value: $val");
end

# ## Question 056
# Generate a generic 2D Gaussian like array with `μ = 0`, `σ = 1` and indices over `{-5, -4, ..., 0, 1, ..., 5}`. (Medium)

vA = -5:5;
μ = 0;
σ = 1;
mG = [(1 / (2 * pi * σ^2)) * exp(-0.5 * (([x, y] .- μ)' * ([x, y] .- μ)) / σ^2) for x in vA, y in vA];

heatmap(mG)

# Using the separability of the Gaussian function:

vG = (1 / (sqrt(2 * pi) * σ)) .* exp.(-0.5 .* (((vA .- μ) .^ 2) / σ^2));
mG = vG * vG';

# Julia 1.10+ using more concise notation:
gauss_2d(x, y, μ=0, σ=1) = exp(-0.5 * ((x-μ)^2 + (y-μ)^2) / σ^2) / (2π * σ^2)
mG_modern = [gauss_2d(x, y) for x in vA, y in vA]

# ## Question 057
# Place `5` elements in a `5x5` array randomly. (Medium)

mA = rand(5, 5);
mA[rand(1:25, 5)] = rand(5);

# Better option which avoids setting into the same indices:
mA = rand(5, 5);
mA[randperm(25)[1:5]] = rand(5);

# Julia 1.10+ using StatsBase.sample (if available):
# using StatsBase
# mA[sample(1:25, 5, replace=false)] = rand(5);

# ## Question 058
# Subtract the mean of each row of a matrix. (Medium)

mA = rand(3, 3);
mA .-= mean(mA, dims = 2);
println("Mean of each column after row normalization:")
mean(mA, dims = 1)

# Julia 1.10+ alternative using broadcasting:
mA = rand(3, 3);
mA_normalized = mA .- mean(mA, dims = 2)

# ## Question 059
# Sort an array by a column. (Medium)

colIdx = 2;

mA = rand(3, 3);
mA[sortperm(mA[:, colIdx]), :]

# Using `sortslices()`:
sortslices(mA, dims = 1, by = x -> x[colIdx]);

# ## Question 060
# Tell if a given 2D array has null (All zeros) columns. (Medium)

mA = rand(0:1, 3, 9);
any(all(iszero.(mA), dims = 1))

# ## Question 061
# Find the 2nd nearest value from a given value in an array. (Medium)

inputVal = 0.5;
vA = rand(10);

vA[sortperm(abs.(vA .- inputVal))[2]]

# Alternative way (More efficient)

closeFirst  = Inf;
closeSecond = Inf;
closeFirstIdx  = 0;
closeSecondIdx = 0;

## Using `global` for scope in Literate
for (elmIdx, elmVal) in enumerate(abs.(vA .- inputVal))
    if (elmVal < closeFirst)
        global closeSecond = closeFirst;
        global closeFirst = elmVal;
        global closeSecondIdx  = closeFirstIdx;
        global closeFirstIdx   = elmIdx;
    elseif (elmVal < closeSecond)
        global closeSecond = elmVal;
        global closeSecondIdx = elmIdx;
    end
end

vA[closeSecondIdx] == vA[sortperm(abs.(vA .- inputVal))[2]]

# By [Tomer Arnon](https://github.com/tomerarnon):

vA[partialsortperm(abs.(vA .- inputVal), 2)]


# ## Question 062
# Considering two arrays with shape `(1, 3)` and `(3, 1)`, Compute their sum using an iterator. (Medium)

vA = rand(1, 3);
vB = rand(3, 1);

sum(aVal + bVal for aVal in vA, bVal in vB)

# ## Question 063
# Create an array class that has a name attribute. (Medium)

# Julia 1.10+ approach using custom struct:
struct NamedArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    name::String
end

function NamedArray(data::Array{T,N}, name::String) where {T,N}
    NamedArray{T,N}(data, name)
end

# Interface methods
Base.size(A::NamedArray) = size(A.data)
Base.getindex(A::NamedArray, i...) = getindex(A.data, i...)
Base.setindex!(A::NamedArray, v, i...) = setindex!(A.data, v, i...)

# Example usage:
named_arr = NamedArray([1 2; 3 4], "My Matrix")

# Alternatively, use NamedArrays.jl or AxisArrays.jl packages

# ## Question 064
# Given a vector, add `1` to each element indexed by a second vector (Be careful with repeated indices). (Hard)

vA = rand(1:10, 5);
vB = rand(1:5, 3);

println(vA);

## Julia is very efficient with loops
for bIdx in vB
    vA[bIdx] += 1;
end

println(vA);

# ## Question 065
# Accumulate elements of a vector `X` to an array `F` based on an index list `I`. (Hard)

vX = rand(1:5, 10);
vI = rand(1:15, 10);

numElements = maximum(vI);
vF = zeros(numElements);

for (ii, iIdx) in enumerate(vI)
    vF[iIdx] += vX[ii];
end

println("vX: $vX");
println("vI: $vI");
println("vF: $vF");

# One could also use `counts()` from `StatsBase.jl`.

# ## Question 066
# Considering an image of size `w x h x 3` image of type `UInt8`, compute the number of unique colors. (Medium)

mI = rand(UInt8, 1000, 1000, 3);

# Method 1: Using reinterpret
numColors = length(unique([reinterpret(UInt32, [iPx[1], iPx[2], iPx[3], 0x00])[1] for iPx in eachrow(reshape(mI, :, 3))]));
println("Number of Unique Colors (Method 1): $numColors");

# Method 2: Using bit operations
numColors = length(unique([UInt32(iPx[1]) + UInt32(iPx[2]) << 8 + UInt32(iPx[3]) << 16 for iPx in eachrow(reshape(mI, :, 3))]));
println("Number of Unique Colors (Method 2): $numColors");

# Method 3: Direct indexing
numColors = length(unique([UInt32(mI[ii, jj, 1]) + UInt32(mI[ii, jj, 2]) << 8 + UInt32(mI[ii, jj, 3]) << 16 for ii in 1:size(mI, 1), jj in 1:size(mI, 2)]));
println("Number of Unique Colors (Method 3): $numColors");

# Julia 1.10+ modern approach using tuples:
color_tuples = Set((mI[i,j,1], mI[i,j,2], mI[i,j,3]) for i in 1:size(mI,1), j in 1:size(mI,2))
numColors_modern = length(color_tuples)
println("Number of Unique Colors (Modern): $numColors_modern");


# ## Question 067
# Considering a four dimensions array, get sum over the last two axis at once. (Hard)

mA = rand(2, 2, 2, 2);
sum(reshape(mA, (2, 2, :)), dims = 3)

# ## Question 068
# Considering a one dimensional vector `vA`, how to compute means of subsets of `vA` using a vector `vS` of same size describing subset indices. (Hard)

# Basically extending Q0065 with another vector of number of additions.

vX = rand(1:5, 10);
vI = rand(1:15, 10);

numElements = maximum(vI);
vF = zeros(numElements);
vN = zeros(Int, numElements);

for (ii, iIdx) in enumerate(vI)
    vF[iIdx] += vX[ii];
    vN[iIdx] += 1;
end

# We only divide by mean if the number of elements accumulated is bigger than 1
for ii in 1:numElements
    vF[ii] = ifelse(vN[ii] > 1, vF[ii] / vN[ii], vF[ii]);
end

println("vX: $vX");
println("vI: $vI");
println("vF: $vF");

# Julia 1.10+ using StatsBase.jl (if available):
# using StatsBase
# means_by_group = [mean(vX[vI .== i]) for i in 1:maximum(vI) if any(vI .== i)]

# ## Question 069
# Get the diagonal of a matrix product. (Hard)

mA = rand(5, 7);
mB = rand(7, 4);

numDiagElements = min(size(mA, 1), size(mB, 2));
vD = [dot(mA[ii, :], mB[:, ii]) for ii in 1:numDiagElements]

# Alternative way:

vD = reshape(sum(mA[1:numDiagElements, :]' .* mB[:, 1:numDiagElements], dims = 1), numDiagElements)

# ## Question 070
# Consider the vector `[1, 2, 3, 4, 5]`, build a new vector with 3 consecutive zeros interleaved between each value. (Hard)

vA = 1:5;

## Since Julia is fast with loops, it would be the easiest choice

numElements = (4 * length(vA)) - 3;
vB = zeros(Int, numElements);

for (ii, bIdx) in enumerate(1:4:numElements)
    vB[bIdx] = vA[ii];
end
println(vB);

## Alternative (MATLAB style) way:

mB = [reshape(collect(vA), 1, :); zeros(Int, 3, length(vA))];
vB = reshape(mB[1:(end - 3)], :);
println(vB);

# ## Question 071
# Consider an array of dimension `5 x 5 x 3`, multiply it by an array with dimensions `5 x 5` using broadcasting. (Hard)

mA = rand(5, 5, 3);
mB = rand(5, 5);

mA .* mB # Very easy in Julia with broadcasting

# Julia 1.10+ explicit demonstration:
result = similar(mA)
for k in 1:size(mA, 3)
    result[:, :, k] = mA[:, :, k] .* mB
end

# ## Question 072
# Swap two rows of a 2D array. (Hard)

mA = rand(UInt8, 3, 2);
println(mA);
mA[[1, 2], :] .= mA[[2, 1], :];
println(mA);

# ## Question 073
# Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles. (Hard)

mA = rand(0:100, 10, 3); # Each row composes 3 vertices ([1] -> [2], [2] -> [3], [3] -> [1])
mC = [sort!([vC[mod1(ii, end)], vC[mod1(ii + 1, end)]]) for ii in 1:(size(mA, 2) + 1), vC in eachrow(mA)][:]
mC = unique(mC)

# Julia 1.10+ cleaner approach:
function get_triangle_edges(triangles)
    edges = Set{Vector{Int}}()
    for triangle in eachrow(triangles)
        for i in 1:3
            edge = sort([triangle[i], triangle[mod1(i+1, 3)]])
            push!(edges, edge)
        end
    end
    return collect(edges)
end

unique_edges = get_triangle_edges(mA)

# ## Question 074
# Given a sorted array `vC` that corresponds to a bincount, produce an array `vA` such that `bincount(vA) == vC`. (Hard)

vC = rand(0:7, 5);
numElements = sum(vC);
vA = zeros(Int, numElements);

elmIdx = 1;
## Using `global` for scope in Literate
for (ii, binCount) in enumerate(vC)
    for jj in 1:binCount
        vA[elmIdx] = ii;
        global elmIdx += 1;
    end
end

# ## Question 075
# Compute averages using a sliding window over an array. (Hard)

numElements = 10;
winRadius   = 1;
winReach    = 2 * winRadius;
winLength   = 1 + winReach;

vA = rand(0:3, numElements);
vB = zeros(numElements - (2 * winRadius));

aIdx = 1 + winRadius;
# Using `global` for scope in Literate
for ii in 1:length(vB)
    vB[ii] = mean(vA[(aIdx - winRadius):(aIdx + winRadius)]); # Using integral / running sum would be faster.
    global aIdx += 1;
end

# Another method using running sum:

vC = zeros(numElements - winReach);

jj = 1;
sumVal = sum(vA[1:winLength]);
vC[jj] = sumVal / winLength;
jj += 1;

# Using `global` for scope in Literate
for ii in 2:(numElements - winReach)
    global sumVal += vA[ii + winReach] - vA[ii - 1];
    vC[jj] = sumVal / winLength;
    global jj += 1;
end

maximum(abs.(vC - vB)) < 1e-8

# Julia 1.10+ modern approach using a function:
function sliding_window_mean(data, window_size)
    n = length(data)
    result = zeros(n - window_size + 1)
    for i in 1:(n - window_size + 1)
        result[i] = mean(@view data[i:(i + window_size - 1)])
    end
    return result
end

vD_modern = sliding_window_mean(vA, winLength)

# ## Question 076
# Consider a one dimensional array `vA`, build a two dimensional array whose first row is `[ vA[1], vA[2], vA[3] ]` and each subsequent row is shifted by 1. (Hard)

vA = rand(10);
numCols = 3;

numRows = length(vA) - numCols + 1;
mA = zeros(numRows, numCols);

for ii in 1:numRows
    mA[ii, :] = vA[ii:(ii + numCols - 1)]; # One could optimize the `-1` out
end

# Julia 1.10+ using stack and broadcasting:
function hankel_matrix(v, cols)
    rows = length(v) - cols + 1
    return [v[i+j-1] for i in 1:rows, j in 1:cols]
end

mA_modern = hankel_matrix(vA, numCols)

# ## Question 077
# Negate a boolean array or change the sign of a float array in-place. (Hard)

vA = rand(Bool, 10);
vA .= .!vA;

vA = randn(10);
vA .*= -1;

# Julia 1.10+ alternatives:
vB_bool = rand(Bool, 10);
map!(!, vB_bool, vB_bool)  # In-place negation

vB_float = randn(10);
@. vB_float = -vB_float    # Using @. macro

# ## Question 078
# Consider 2 sets of points `mP1`, `mP2` describing lines (2d) and a point `vP`, how to compute distance from the point `vP` to each line `i`: `[mP1[i, :], mP2[i, :]]`. (Hard)

## See distance of a point from a line in Wikipedia (https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line).  
## Specifically _Line Defined by Two Points_.

numLines = 10;
mP1 = randn(numLines, 2);
mP2 = randn(numLines, 2);
vP  = randn(2);

# Modern Julia 1.10+ approach:
function point_to_line_distance(p1, p2, point)
    # Distance from point to line defined by p1 and p2
    return abs((p2[1] - p1[1]) * (p1[2] - point[2]) - (p1[1] - point[1]) * (p2[2] - p1[2])) / 
           hypot(p2[1] - p1[1], p2[2] - p1[2])
end

vD = [point_to_line_distance(mP1[i,:], mP2[i,:], vP) for i in 1:numLines];
# Alternative using eachrow:
vD_alt = [point_to_line_distance(p1, p2, vP) for (p1, p2) in zip(eachrow(mP1), eachrow(mP2))];

minDist = minimum(vD);
println("Min Distance: $minDist");

# ## Question 079
# Consider 2 sets of points `mP1`, `mP2` describing lines (2d) and a set of points `mP`, how to compute distance from the point `vP = mP[j, :]` to each line `i`: `[mP1[i, :], mP2[i, :]]`. (Hard)

numLines = 5;
mP1 = randn(numLines, 2);
mP2 = randn(numLines, 2);
mP  = randn(numLines, 2);

# Modern Julia 1.10+ approach using broadcasting:
function distance_matrix(lines_p1, lines_p2, points)
    function line_point_dist(p1, p2, point)
        return abs((p2[1] - p1[1]) * (p1[2] - point[2]) - (p1[1] - point[1]) * (p2[2] - p1[2])) / 
               hypot(p2[1] - p1[1], p2[2] - p1[2])
    end
    
    return [line_point_dist(lines_p1[i,:], lines_p2[i,:], points[j,:]) 
            for i in 1:size(lines_p1,1), j in 1:size(points,1)]
end

mD = distance_matrix(mP1, mP2, mP);

for jj in 1:numLines
    minDist = minimum(mD[jj, :]);
    println("The minimum distance from the $jj -th point: $minDist");
end

# ## Question 080
# Consider an arbitrary 2D array, write a function that extracts a subpart with a fixed shape and centered on a given element (Handle out of bounds). (Hard)

## One could use `PaddedViews.jl` to easily solve this.

arrayLength = 10;
winRadius   = 3;
vWinCenter  = [7, 9];

mA = rand(arrayLength, arrayLength);
winLength = (2 * winRadius) + 1;

# Modern Julia 1.10+ approach:
function extract_subarray(array, center, radius)
    win_size = 2 * radius + 1
    result = zeros(eltype(array), win_size, win_size)
    
    for i in 1:win_size
        for j in 1:win_size
            row_idx = center[1] + i - radius - 1
            col_idx = center[2] + j - radius - 1
            # Use clamping for nearest neighbor extrapolation
            safe_row = clamp(row_idx, 1, size(array, 1))
            safe_col = clamp(col_idx, 1, size(array, 2))
            result[i, j] = array[safe_row, safe_col]
        end
    end
    return result
end

mB = extract_subarray(mA, vWinCenter, winRadius);

# Alternative using comprehension:
mB_alt = [mA[clamp(vWinCenter[1] + i - winRadius - 1, 1, arrayLength), 
             clamp(vWinCenter[2] + j - winRadius - 1, 1, arrayLength)] 
          for i in 1:(2*winRadius+1), j in 1:(2*winRadius+1)];

# ## Question 081
# Consider an array `vA = [1, 2, 3, ..., 13, 14]`, generate an array `vB = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], ..., [11, 12, 13, 14]]`. (Hard)

vA = collect(1:14);

winNumElements  = 4;
winReach        = winNumElements - 1;

# Modern Julia 1.10+ sliding window approach:
vB = [vA[ii:(ii + winReach)] for ii in 1:(length(vA) - winReach)];

# Alternative using partition from IterTools.jl:
# using IterTools
# vB_alt = collect(partition(vA, winNumElements, 1));

# Using views for memory efficiency:
vB_views = [@view vA[i:(i + winReach)] for i in 1:(length(vA) - winReach)];

# ## Question 082
# Compute a matrix rank. (Hard)

numRows = 5;
numCols = 4;
mA = randn(numRows, numCols);

# Modern Julia 1.10+ approaches:
matrix_rank = rank(mA);
println("Matrix rank: $matrix_rank");

# Alternative with tolerance specification:
matrix_rank_tol = rank(mA, atol=1e-10);

# Using LinearAlgebra functions for more control:
using LinearAlgebra
svd_result = svd(mA);
rank_svd = count(x -> x > 1e-10, svd_result.S);  # Count non-zero singular values

# ## Question 083
# Find the most frequent value in an array. (Hard)

vA = rand(1:5, 15);

# Modern Julia 1.10+ approaches:

# Method 1: Using StatsBase.jl (most efficient)
# using StatsBase
# most_frequent = mode(vA)

# Method 2: Using countmap for detailed frequency analysis
# using StatsBase
# freq_map = countmap(vA)
# most_frequent = findmax(freq_map)[2]

# Method 3: Manual implementation (MATLAB style)
vB = unique(vA);
frequencies = [count(==(val), vA) for val in vB];
most_frequent = vB[argmax(frequencies)];

# Method 4: Broadcasting approach (original style, improved)
freq_counts = dropdims(sum(vA .== vB', dims=1), dims=1);
most_frequent_broadcast = vB[argmax(freq_counts)];

println("Most frequent value: $most_frequent");

# Comparing bits:

# One could convert at the bits level to integers and then use something like `counts()` from `StatsBase.jl`.
# Support to 1:4 bytes of data:
# ```julia
# numBytes = sizeof(vA[1]);
# if (sizeof(vA[1]) == 1)
#     vB = reinterpret(UInt8, vA);
# elseif (sizeof(vA[1]) == 2)
#     vB = reinterpret(UInt16, vA);
# elseif (sizeof(vA[1]) == 4)
#     vB = reinterpret(UInt32, vA);
# elseif (sizeof(vA[1]) == 8)
#     vB = reinterpret(UInt64, vA);
# end
# ```

# ## Question 084
# Extract all the contiguous `3x3` blocks from a random `5x5` matrix. (Hard)

numRows = 5;
numCols = 5;

mA = rand(1:9, numRows, numCols);

winRadius   = 1;
winReach    = 2 * winRadius;
winLength   = winReach + 1;

# Modern Julia 1.10+ approach with views for memory efficiency:
mB = [@view mA[ii:(ii + winReach), jj:(jj + winReach)] 
      for ii in 1:(numRows - winReach), jj in 1:(numCols - winReach)];

# Alternative: copying the blocks
mB_copy = [mA[ii:(ii + winReach), jj:(jj + winReach)] 
           for ii in 1:(numRows - winReach), jj in 1:(numCols - winReach)];

# Using comprehension for 3x3 blocks specifically:
blocks_3x3 = [mA[i:i+2, j:j+2] for i in 1:3, j in 1:3];

# ## Question 085
# Create a 2D array struct such that `mA[i, j] == mA[j, i]` (Symmetric matrix). (Hard)

# Modern Julia 1.10+ approach using LinearAlgebra.Symmetric:
using LinearAlgebra

# Method 1: Using built-in Symmetric wrapper
mA_data = randn(4, 4);
mA_symmetric = Symmetric(mA_data);  # Automatically ensures symmetry

# Method 2: Custom symmetric matrix struct
struct SymmetricMatrix{T <: Number} <: AbstractArray{T, 2}
    size::Int
    data::Vector{T}  # Store only upper triangle
    
    function SymmetricMatrix{T}(n::Int) where T
        new{T}(n, zeros(T, div(n * (n + 1), 2)))
    end
    
    function SymmetricMatrix(matrix::AbstractMatrix{T}) where T
        n = size(matrix, 1)
        size(matrix, 2) == n || throw(ArgumentError("Matrix must be square"))
        sym_mat = SymmetricMatrix{T}(n)
        for i in 1:n, j in i:n
            sym_mat[i, j] = (matrix[i, j] + matrix[j, i]) / 2
        end
        return sym_mat
    end
end

# Define required AbstractArray interface
Base.size(A::SymmetricMatrix) = (A.size, A.size)
Base.IndexStyle(::Type{<:SymmetricMatrix}) = IndexCartesian()

function Base.getindex(S::SymmetricMatrix, i::Int, j::Int)
    i, j = minmax(i, j)  # Ensure i <= j
    idx = div((j - 1) * j, 2) + i
    return S.data[idx]
end

function Base.setindex!(S::SymmetricMatrix{T}, val::T, i::Int, j::Int) where T
    i, j = minmax(i, j)
    idx = div((j - 1) * j, 2) + i
    S.data[idx] = val
    return val
end

# Test the symmetric matrix
test_matrix = randn(3, 3);
sym_test = SymmetricMatrix(test_matrix);
println("sym_test[1,2] == sym_test[2,1]: $(sym_test[1,2] == sym_test[2,1])");

function Base.size(mA::SymmetricMatrix)
    (mA.numRows, mA.numRows);
end
function Base.getindex(mA::SymmetricMatrix, ii::Int)
    mA.data[ii];
end
function Base.getindex(mA::SymmetricMatrix, ii::Int, jj::Int)
    mA.data[ii, jj];
end
function Base.setindex!(mA::SymmetricMatrix, v, ii::Int, jj::Int) 
    setindex!(mA.data, v, ii, jj);
    setindex!(mA.data, v, jj, ii);
# Test the symmetric matrix
test_matrix = randn(3, 3);
sym_test = SymmetricMatrix(test_matrix);
println("sym_test[1,2] == sym_test[2,1]: $(sym_test[1,2] == sym_test[2,1])");

# Example usage
mA = SymmetricMatrix{Int}(2);
mA[1, 2] = 5;
println("Matrix symmetry: mA[1,2] = $(mA[1,2]), mA[2,1] = $(mA[2,1])");

# ## Question 086
# Consider a set of `p` matrices of shape `nxn` and a set of `p` vectors with length `n`. Compute the sum of the `p` matrix vector products at once (Result is a vector of length `n`). (Hard)

## One could use `TensorOperations.jl` or `Einsum.jl` for a more elegant solution.

numRows = 5;
numMat  = 3;

tP = [randn(numRows, numRows) for _ in 1:numMat];
mP = [randn(numRows) for _ in 1:numMat];

# Modern Julia 1.10+ approaches:

# Method 1: Using reduce with broadcasting
vA = reduce(+, (matrix * vector for (matrix, vector) in zip(tP, mP)));

# Method 2: Using sum with generator
vA_v2 = sum(matrix * vector for (matrix, vector) in zip(tP, mP));

# Method 3: Manual accumulation (for educational purposes)
vA_manual = zeros(numRows);
for (matrix, vector) in zip(tP, mP)
    vA_manual .+= matrix * vector
end

# Method 4: Using mapreduce
vA_v3 = mapreduce(Base.splat(*), +, zip(tP, mP));

# Original vanilla solution
vB = zeros(numRows);
for ii in 1:numMat
    vB .+= tP[ii] * mP[ii];
end

@assert vA ≈ vB  # Verify results are equivalent

# ## Question 087
# Consider a `16x16` array, calculate the block sum (Block size is `4x4`). (Hard)

# We solve a more general case for any size of blocks.

numRows = 16;
numCols = 8;

vBlockSize = [2, 4]; #<! [numRows, numCols] ./ vBlockSize == integer

mA = rand(numRows, numCols);

numBlocksVert = numRows ÷ vBlockSize[1];
numBlocksHori = numCols ÷ vBlockSize[2];
numBlocks = numBlocksVert * numBlocksHori;

# Modern Julia 1.10+ approach using reshape and sum:
function block_sum(matrix, block_size)
    rows, cols = size(matrix)
    block_rows, block_cols = block_size
    
    # Ensure dimensions are compatible
    @assert rows % block_rows == 0 && cols % block_cols == 0
    
    n_blocks_vert = rows ÷ block_rows
    n_blocks_hori = cols ÷ block_cols
    
    # Reshape and sum over blocks
    reshaped = reshape(matrix, block_rows, n_blocks_vert, block_cols, n_blocks_hori)
    block_sums = dropdims(sum(reshaped, dims=(1,3)), dims=(1,3))
    
    return block_sums'  # Transpose to match expected output format
end

mB_blocks = block_sum(mA, vBlockSize);

# Alternative approach using comprehension:
mB_alt = [sum(mA[(i-1)*vBlockSize[1]+1:i*vBlockSize[1], 
                 (j-1)*vBlockSize[2]+1:j*vBlockSize[2]]) 
          for i in 1:numBlocksVert, j in 1:numBlocksHori];

# ## Question 088
# Implement the simulation _Game of Life_ using arrays. (Hard)

numRows = 20;
numCols = 20;

# Modern Julia 1.10+ Game of Life implementation:
function game_of_life_step!(current::AbstractMatrix{Bool}, next::AbstractMatrix{Bool})
    fill!(next, false)
    rows, cols = size(current)
    
    for i in 1:rows
        for j in 1:cols
            # Count living neighbors
            neighbors = 0
            for di in -1:1, dj in -1:1
                if di == 0 && dj == 0
                    continue
                end
                
                ni, nj = i + di, j + dj
                if 1 <= ni <= rows && 1 <= nj <= cols
                    neighbors += current[ni, nj]
                end
            end
            
            # Apply Game of Life rules
            if current[i, j]  # Cell is alive
                next[i, j] = neighbors == 2 || neighbors == 3
            else  # Cell is dead
                next[i, j] = neighbors == 3
            end
        end
    end
end

# Initialize with random living cells
gofNumLives = round(Int, 0.05 * numRows * numCols);
gofNumGenerations = 50;

vI = randperm(numRows * numCols)[1:gofNumLives];
mG = falses(numRows, numCols);
mG[vI] .= true;
mB = similar(mG);

println("Initial state:");
display(mG);

# Run simulation
for generation in 1:gofNumGenerations
    game_of_life_step!(mG, mB)
    mG, mB = mB, mG  # Swap buffers
end

println("Final state after $gofNumGenerations generations:");
display(mG);

# ## Question 089
# Get the `n` largest values of an array. (Hard)

vA = rand(10);
numValues = 3;

# Modern Julia 1.10+ approaches:

# Method 1: Using partialsortperm (most efficient for large arrays)
largest_values = vA[partialsortperm(vA, 1:numValues, rev=true)];

# Method 2: Using sort and indexing (simple but less efficient)
largest_values_v2 = sort(vA, rev=true)[1:numValues];

# Method 3: Using findmax repeatedly (good for very small n)
function get_n_largest(arr, n)
    temp = copy(arr)
    results = similar(arr, n)
    for i in 1:n
        idx = argmax(temp)
        results[i] = temp[idx]
        temp[idx] = -Inf  # Or minimum value
    end
    return results
end

largest_values_v3 = get_n_largest(vA, numValues);

println("Largest $numValues values: $largest_values");

# ## Question 090
# Given an arbitrary number of vectors, build the _Cartesian Product_ (Every combination of every item). (Hard)

# Modern Julia 1.10+ approaches:

# Method 1: Using Iterators.product (most efficient)
function cartesian_product(vectors...)
    return collect(Iterators.product(vectors...))
end

# Method 2: Flatten the result for easy access
function cartesian_product_flat(vectors...)
    return vec(collect(Iterators.product(vectors...)))
end

vA = 1:3;
vB = 8:9;
vC = 4:5;

cart_prod = cartesian_product(vA, vB, vC);
cart_prod_flat = cartesian_product_flat(vA, vB, vC);

println("Cartesian product (matrix form):");
display(cart_prod);
println("Cartesian product (flat):");
display(cart_prod_flat);

# ## Question 091
# Create an array which can be accessed like a _record array_ in _NumPy_. (Hard)

# Modern Julia 1.10+ approaches:

# Method 1: Using NamedTuple of arrays (most Julia-like)
function create_record_array(; kwargs...)
    # All arrays must have the same length
    lengths = [length(v) for v in values(kwargs)]
    @assert all(==(first(lengths)), lengths) "All arrays must have same length"
    
    return NamedTuple(kwargs)
end

# Method 2: Using struct with custom indexing
struct RecordArray{T<:NamedTuple}
    data::T
    length::Int
    
    function RecordArray(; kwargs...)
        nt = NamedTuple(kwargs)
        lengths = [length(v) for v in values(nt)]
        @assert all(==(first(lengths)), lengths) "All arrays must have same length"
        new{typeof(nt)}(nt, first(lengths))
    end
end

Base.length(ra::RecordArray) = ra.length
Base.getindex(ra::RecordArray, i::Int) = NamedTuple{keys(ra.data)}(getindex.(values(ra.data), i))
Base.getindex(ra::RecordArray, field::Symbol) = getproperty(ra.data, field)

# Example usage
names = ["Alice", "Bob", "Charlie"];
ages = [25, 30, 35];
scores = [85.5, 92.0, 78.5];

# Method 1: NamedTuple approach
record_nt = create_record_array(name=names, age=ages, score=scores);
println("Record 1 (NamedTuple): $(record_nt.name[1]), $(record_nt.age[1]), $(record_nt.score[1])");

# Method 2: Custom struct approach
record_array = RecordArray(name=names, age=ages, score=scores);
println("Record 1 (Struct): $(record_array[1])");
println("All names: $(record_array[:name])");

# ## Question 092
# Consider a large vector `vA`, compute `vA` to the power of 3 using 3 different methods. (Hard)

vA = rand(1000);

# ## Question 092
# Consider a large vector `vA`, compute `vA` to the power of 3 using 3 different methods. (Hard)

vA = rand(1000);

# Modern Julia 1.10+ approaches:

# Method 1: Broadcasting with power operator (most concise)
vB = vA .^ 3;

# Method 2: Using comprehension
vC = [valA ^ 3 for valA in vA];

# Method 3: Manual loop with pre-allocation
vD = zeros(length(vA));
for (ii, valA) in enumerate(vA)
    vD[ii] = valA * valA * valA;
end

# Method 4: Broadcasting with repeated multiplication (potentially faster)
vE = vA .* vA .* vA;

# Method 5: Using map function
vF = map(x -> x^3, vA);

# Verify all methods give same result
@assert vB ≈ vC ≈ vD ≈ vE ≈ vF

println("All methods produce equivalent results");

# ## Question 093
# Consider two arrays `mA` and `mB` of shape `8x3` and `2x2`. Find rows of `mA` that contain elements of each row of `mB` regardless of the order of the elements in `mB`. (Hard)

# The way I interpret the question is rows in `mA` which contain at least 1 element from each row of `mB`.

mA = rand(0:4, 8, 3);
mB = rand(0:4, 2, 2);

# Modern Julia 1.10+ approach using broadcasting and any/all:
function find_containing_rows(matrix_a, matrix_b)
    # For each row in mB, check which rows in mA contain any of its elements
    containment_matrix = [any(row_a .== row_b') for row_b in eachrow(matrix_b), row_a in eachrow(matrix_a)]
    
    # Find rows in mA that contain elements from ALL rows in mB
    return [all(col) for col in eachcol(containment_matrix)]
end

vD = find_containing_rows(mA, mB);

# Alternative using Set operations for better performance with larger arrays:
function find_containing_rows_sets(matrix_a, matrix_b)
    set_b_rows = [Set(row) for row in eachrow(matrix_b)]
    
    return [all(any(val in set_b for val in row_a) for set_b in set_b_rows) 
            for row_a in eachrow(matrix_a)]
end

vD_sets = find_containing_rows_sets(mA, mB);

println("Rows in mA containing elements from each row of mB: ", findall(vD));

# ## Question 094
# Considering a `10x3` matrix, extract rows with unequal values. (Medium)

mA = rand(1:3, 10, 3);

# Modern Julia 1.10+ approaches:

# Method 1: Using extrema for efficiency
vD1 = [extrema(row)[1] != extrema(row)[2] for row in eachrow(mA)];

# Method 2: Using allequal (available in Julia 1.8+)
vD2 = [!allequal(row) for row in eachrow(mA)];

# Method 3: Original approach using string representation
vD3 = [maximum(vA) != minimum(vA) for vA in eachrow(mA)];

# Method 4: Using Set for unique values
vD4 = [length(Set(row)) > 1 for row in eachrow(mA)];

# Extract rows with unequal values
unequal_rows = mA[vD2, :];

println("Number of rows with unequal values: $(sum(vD2))");

# ## Question 095
# Convert a vector of ints into a matrix binary representation. (Hard)

vA = rand(UInt8, 10);

# Modern Julia 1.10+ approaches:

# Method 1: Using digits function (most elegant)
mB_digits = reverse!(reduce(hcat, digits.(vA, base=2, pad=8))', dims=2);

# Method 2: Using bitstring and parsing (educational)
function int_to_binary_matrix(int_vector, num_bits=8)
    n = length(int_vector)
    binary_matrix = falses(n, num_bits)
    
    for (i, val) in enumerate(int_vector)
        bit_str = bitstring(val)[end-num_bits+1:end]  # Take last num_bits
        for (j, bit_char) in enumerate(bit_str)
            binary_matrix[i, j] = bit_char == '1'
        end
    end
    
    return binary_matrix
end

mB_bitstring = int_to_binary_matrix(vA);

# Method 3: Using bit operations (most efficient)
function int_to_binary_fast(int_vector, num_bits=8)
    n = length(int_vector)
    binary_matrix = falses(n, num_bits)
    
    for (i, val) in enumerate(int_vector)
        for j in 1:num_bits
            binary_matrix[i, j] = (val >> (num_bits - j)) & 1 == 1
        end
    end
    
    return binary_matrix
end

mB_fast = int_to_binary_fast(vA);

# Verify all methods give same result
@assert mB_digits == mB_bitstring == mB_fast

println("Binary representation of first integer ($(vA[1])):");
println(mB_digits[1, :]);

# ## Question 096
# Given a two dimensional array, extract unique rows. (Hard)

mA = UInt8.(rand(1:3, 10, 3));

# Modern Julia 1.10+ approaches:

# Method 1: Using unique directly on eachrow (most elegant)
unique_rows_v1 = unique(eachrow(mA));
mA_unique_v1 = reduce(vcat, reshape.(unique_rows_v1, 1, :));

# Method 2: Using Set for row comparison
function get_unique_rows(matrix)
    seen_rows = Set{Vector{eltype(matrix)}}()
    unique_indices = Int[]
    
    for (i, row) in enumerate(eachrow(matrix))
        row_vec = collect(row)
        if !(row_vec in seen_rows)
            push!(seen_rows, row_vec)
            push!(unique_indices, i)
        end
    end
    
    return matrix[unique_indices, :]
end

mA_unique_v2 = get_unique_rows(mA);

# Method 3: Original approach using string representation
vS = [reduce(*, bitstring(valA) for valA in vA) for valA
vU = unique(vS);
vI = [findfirst(valU .== vS) for valU in vU];
mA_unique_v3 = mA[vI, :];

# Method 4: Using indexing with unique
vB = indexin(vU, vS);
@assert vB == vI

println("Original matrix size: $(size(mA))");
println("Unique rows matrix size: $(size(mA_unique_v1))");

# ## Question 097
# Considering 2 vectors `vA` and `vB`, write the einsum equivalent (Using `Einsum.jl`) of inner, outer, sum, and mul function. (Hard)

vA = rand(5);
vB = rand(5);

# Modern Julia 1.10+ approaches using Tullio.jl (successor to Einsum.jl):

# Method 1: Using Tullio.jl (recommended)
# using Tullio

# Inner Product
# @tullio inner_result = vA[i] * vB[i]
# @assert inner_result ≈ dot(vA, vB)

# Outer Product  
# @tullio outer_result[i, j] := vA[i] * vB[j]
# @assert outer_result ≈ vA * vB'

# Sum
# @tullio sum_result = vA[i]
# @assert sum_result ≈ sum(vA)

# Element-wise multiplication
# @tullio mult_result[i] := vA[i] * vB[i]
# @assert mult_result ≈ vA .* vB

# Method 2: Manual implementations showing the operations
inner_manual = sum(vA .* vB);  # Inner product
outer_manual = vA * vB';       # Outer product
sum_manual = sum(vA);          # Sum
mult_manual = vA .* vB;        # Element-wise multiplication

println("Inner product: $inner_manual");
println("Sum: $sum_manual");
println("Outer product size: $(size(outer_manual))");
println("Element-wise multiplication: $mult_manual");

# ## Question 098
# Considering a path described by two vectors `vX` and `vY`, sample it using equidistant samples. (Hard)

# The way I interpreted the question is to create sub segments of the same length.  

numPts = 100;
numSegments = 1000;

vX = sort(10 * rand(numPts));
vY = sort(10 * rand(numPts));

# Modern Julia 1.10+ approach:

# Calculate cumulative distances along the path
vR = cumsum([0.0; hypot.(diff(vX), diff(vY))]);
vRSegment = range(0.0, vR[end], length=numSegments);

# Create a simple linear interpolation function
function linear_interpolate(x_vals, y_vals, target_x)
    if target_x <= x_vals[1]
        return y_vals[1]
    elseif target_x >= x_vals[end]
        return y_vals[end]
    else
        # Find bracketing indices
        right_idx = findfirst(x >= target_x for x in x_vals)
        left_idx = right_idx - 1
        
        # Linear interpolation
        t = (target_x - x_vals[left_idx]) / (x_vals[right_idx] - x_vals[left_idx])
        return (1 - t) * y_vals[left_idx] + t * y_vals[right_idx]
    end
end

# Sample the path at equidistant points
vXSegment = [linear_interpolate(vR, vX, r) for r in vRSegment];
vYSegment = [linear_interpolate(vR, vY, r) for r in vRSegment];

println("Original path points: $numPts");
println("Resampled path points: $numSegments");
println("Total path length: $(vR[end])");

# ## Question 099
# Given an integer `n` and a 2D array `mA`, find the rows which can be interpreted as draws from a multinomial distribution with `n` (Rows which only contain integers and which sum to `n`). (Hard)

mA = rand([0, 0.5, 1, 2, 3], 15, 3);
sumVal = 4;

# Modern Julia 1.10+ approach:
function find_multinomial_rows(matrix, target_sum)
    return [all(row .== round.(row)) && sum(row) == target_sum for row in eachrow(matrix)]
end

vI = find_multinomial_rows(mA, sumVal);

# Alternative using isinteger for clarity:
vI_alt = [all(isinteger, row) && sum(row) == sumVal for row in eachrow(mA)];

# Get the actual multinomial rows
multinomial_rows = mA[vI, :];

println("Number of multinomial rows found: $(sum(vI))");
if sum(vI) > 0
    println("Example multinomial row: $(multinomial_rows[1, :])");
end

# ## Question 100
# Compute bootstrapped `95%` confidence intervals for the mean of a 1D array `vA`. Namely, resample the elements of an array with replacement `N` times, compute the mean of each sample and then compute percentiles over the means. (Hard)

numTrials = 10000;
numSamples = 1000;
μ = 0.5;

vA = μ .+ randn(numSamples);

# Modern Julia 1.10+ bootstrap approach:

# Method 1: Using generator expression and StatsBase.jl
# using StatsBase
# bootstrap_means = [mean(sample(vA, numSamples, replace=true)) for _ in 1:numTrials]
# confidence_interval = quantile(bootstrap_means, [0.025, 0.975])

# Method 2: Manual bootstrap with random sampling
function bootstrap_confidence_interval(data, n_bootstrap=10000, confidence_level=0.95)
    n_data = length(data)
    bootstrap_means = Vector{Float64}(undef, n_bootstrap)
    
    for i in 1:n_bootstrap
        # Sample with replacement
        bootstrap_sample = data[rand(1:n_data, n_data)]
        bootstrap_means[i] = mean(bootstrap_sample)
    end
    
    # Calculate confidence interval
    α = 1 - confidence_level
    lower_percentile = 100 * (α / 2)
    upper_percentile = 100 * (1 - α / 2)
    
    return quantile(bootstrap_means, [α/2, 1-α/2])
end

# Calculate confidence interval
conf_interval = bootstrap_confidence_interval(vA, numTrials);
println("Bootstrap 95% confidence interval for the mean: $(conf_interval)");
println("True mean: $μ");
println("Sample mean: $(mean(vA))");

# Original approach (more compact)
tM = (mean(vA[rand(1:numSamples, numSamples)]) for _ in 1:numTrials);
conf_interval_original = quantile(tM, [0.025, 0.975]);
println("Original method confidence interval: $(conf_interval_original)");
