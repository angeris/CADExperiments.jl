using Test
using CADConstraints

@testset "module loads" begin
    @test isdefined(CADConstraints, :SparseLNNS)
end
