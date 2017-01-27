module NetSci02

using LightGraphs, GraphPlot
using Distributions
using PyPlot

include("utils.jl")


"""
    samplepercolations(measure::Function, graph::Graph, repetitions::Integer,
                       node_distributions::Vector{F}, range)

Sample `repetitions` node removals on `graph`, calculating `measure` each time.  The
probability of deleting a nodes is calculated by `node_distribution(φ)` for each φ.
"""
function samplepercolations{F<:Function}(measure::Function,
                                         graph::Graph,
                                         repetitions::Integer,
                                         node_distributions::Vector{F},
                                         range = 0.1:0.1:1.0,
                                         result_type = typeof(measure(Graph(1, 0))))
    @assert all(map(≈, extrema(range), (0.0, 1.0)))
    
    const vertices = nv(graph)
    curves = SharedArray(result_type, length(range), repetitions)
    
    @sync for (i, φ) in enumerate(range)
        @parallel for r in 1:repetitions
            victim = copy(graph)
            for v = vertices:-1:1
                if rand(node_distributions[v](φ)) == 1
                    rem_vertex!(victim, v)
                end
            end
            
            curves[i, r] = measure(victim)
        end
    end
    
    return curves
end


# """
#     informed_probabilities(graph::Graph, weights::Vector{Float64})

# Construct a distribution over nodes which integrates knowledge about local clustering,
# degree, and second-order neighbourhood, taking into account `weights`.
# """
# function informed_probabilities(graph::Graph, weights::Vector{Float64},
#                                 infos::Vector{Symbol})::Vector{Bernoulli}
#     const maxdegrees = nv(graph) * (nv(graph) + 1) / 2
#     const information_functions = Dict(
#         :local_clustering => g -> local_clustering_coefficient(g),
#         :degree => g -> degree(g) / maxdegrees,
#         :neighbourhood2 => g -> map(v -> length(neighborhood(g, v, 2)), vertices(g)) / maxdegrees,
#         :neighbourhood3 => g -> map(v -> length(neighborhood(g, v, 3)), vertices(g)) / maxdegrees
#     )
    
#     information = Array(Float64, nv(graph), length(infos))
#     for (i, info) in enumerate(infos)
#         information[:, i] = information_functions[info](graph)
#     end

#     Bernoulli.(logistic.(information * weights))
# end


# """
#     percolation_loss(graph, samples)

# Construct a loss function which samples the expected area of the percolation curve of `graph`,
# using the parameter as weightings.
# """
# function percolation_loss(graph::Graph, samples::Integer, infos::Vector{Symbol})
#     function loss(parameter)
#         dist(g) = informed_probabilities(g, parameter, infos)
#         curves = samplepercolations(graph, samples, dist) do g
#             maximum(length(c) for c in connected_components(g))
#         end

#         normalizers = mapslices(maximum, curves, 2)
#         # every curve is normed relative to its maximum (highest) value; then, average
#         sum(curves ./ normalizers) / samples
#     end
# end




function simulate_random(;vertices = 1000, edges = 10000)
    graph = Graph(vertices, edges)
    range = 0.0:0.1:1.0
    
    results = samplepercolations(graph, 10, fill(φ -> Bernoulli(φ), vertices), range) do g
        reduce(max, 0, length.(connected_components(g)))
    end

    plot(range, results)
end


# function test(;nv = 100, ne = 50)
#     graph = Graph(nv, ne)
#     # samplepercolations(graph, 10, _ -> DiscreteUniform(1, nv)) do g
#     #     length(connected_components(g)), length(triangles(g))
#     # end

#     error_trace = Float64[]
    
#     function log_annealing(;kwargs...)
#         args = Dict(kwargs)
#         push!(error_trace, args[:f_new])
        
#         if args[:steps] % 10 == 0
#             println(args[:steps], " steps, T = ", args[:T])
#             println("\t", args[:x_new])
#             println("\t", args[:f_new])
#         end
#     end

#     infos = [:degree]
#     annealing(percolation_loss(graph, 30, infos), randn(length(infos)),
#               1.0, 0.99, 0.05, (s, v) -> s > 1000;
#               debug_callback = log_annealing)

#     plot(error_trace)
# end

end
