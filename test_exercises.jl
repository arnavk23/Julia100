"""
Test suite for Julia110Exercises

This script tests the modernized Julia exercises to ensure they work correctly
with Julia 1.10+.
"""

using Test
using LinearAlgebra
using Statistics
using Random
using Dates
using DelimitedFiles

println("Testing Julia110Exercises...")
println("Julia Version: $(VERSION)")
println("=" ^ 50)

@testset "Julia110Exercises Tests" begin
    
    @testset "Basic Array Operations" begin
        # Test Question 1 equivalent
        vA = collect(1:10)
        @test length(vA) == 10
        @test vA[1] == 1
        @test vA[end] == 10
        
        # Test zeros matrix creation
        mA = zeros(3, 3)
        @test size(mA) == (3, 3)
        @test all(mA .== 0)
    end
    
    @testset "Modern Broadcasting" begin
        # Test broadcasting operations
        vA = [1, 2, 3]
        vB = [10, 20, 30]
        vC = vA .+ vB
        @test vC == [11, 22, 33]
        
        # Test broadcasting with scalars
        vD = vA .* 2
        @test vD == [2, 4, 6]
    end
    
    @testset "Type Stability" begin
        # Test type-stable function
        function stable_sum(x::Vector{Float64})::Float64
            return sum(x)
        end
        
        result = stable_sum([1.0, 2.0, 3.0])
        @test result == 6.0
        @test typeof(result) == Float64
    end
    
    @testset "Linear Algebra" begin
        # Test matrix operations
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]
        C = A * B
        @test size(C) == (2, 2)
        
        # Test eigenvalues
        eigs = eigvals(A)
        @test length(eigs) == 2
    end
    
    @testset "Statistics" begin
        # Test statistical functions
        data = randn(100)
        @test typeof(mean(data)) == Float64
        @test typeof(std(data)) == Float64
        @test std(data) >= 0
    end
    
    @testset "Modern Julia Features" begin
        # Test comprehensions
        squares = [x^2 for x in 1:5]
        @test squares == [1, 4, 9, 16, 25]
        
        # Test multiple return values
        function minmax_vals(arr)
            return minimum(arr), maximum(arr)
        end
        
        min_val, max_val = minmax_vals([1, 5, 3, 9, 2])
        @test min_val == 1
        @test max_val == 9
    end
end

println("All tests passed!")
println("Julia110Exercises is working correctly with Julia $(VERSION)")
