module NetSci02

using LightGraphs, GraphPlot
using Distributions
using PyPlot

include("utils.jl")


"""
    samplepercolations(measure::Function, graph::Graph, node_distributions::Function,
                       repetitions::Integer, stepsize = 0.1)

Sample `repetitions` node removals on `graph`, over deletion rates `0.0:stepsize:1.0`, calculating
`measure` each time.  The probability of deleting a nodes is given by `node_distributions(φ)`
for each φ (so `node_distributions` must have type Float -> Vector{Bernoulli}).
"""
function samplepercolations(measure::Function,
                            graph::Graph,
                            node_distributions::Function,
                            repetitions::Integer,
                            stepsize = 0.1,
                            result_type = typeof(measure(Graph(1, 0))))
    const vertices = nv(graph)
    const range = 0.0:stepsize:1.0
    curves = SharedArray(result_type, repetitions, length(range))
    
    pmap(1:repetitions) do r
        for (i, φ) in enumerate(range)
            victim = copy(graph)
            distributions = node_distributions(φ)
            
            for v = vertices:-1:1
                if rand(distributions[v]) == 1
                    rem_vertex!(victim, v)
                end
            end
            
            curves[r, i] = measure(victim)
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
#                                 infos::Vector{Symbol})::Function
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

#     probabilities = logistic.(information * weights)
#     mean_probability = mean(probabilities)
#     return φ -> Bernoulli.(probabilities - mean_probability + φ)
# end


"""
    percolation_loss(graph, samples)

Construct a loss function which samples the expected area of the percolation curve of `graph`,
using the parameter as weightings.
"""
function percolation_loss(graph::Graph, infos::Vector{Symbol}, samples = 10, stepsize = 0.05)
    function loss(parameter)
        dist(g) = informed_probabilities(g, parameter, infos)
        curves = samplepercolations(graph, samples, dist) do g
            reduce(max, 0, length.(connected_components(g)))
        end

        # normalize by maximum value, then integrate and average
        normalized_curves = curves ./ mapslices(maximum, curves, 2)
        return sum(normalized_curves) * stepsize / samples
    end
end


normalize_curve(curve) = curve / maximum(curve)
percolation_area(curves, stepsize) = sum(curves ./ mapslices(maximum, curves, 2)) * stepsize / size(curves)[2]

function simulate_random(graph = Graph(1000, 2000))
    const stepsize = 0.05
    results = samplepercolations(graph, φ -> fill(Bernoulli(φ), nv(graph)), 10, stepsize) do g
        reduce(max, 0, length.(connected_components(g)))
    end

    println(percolation_area(results, stepsize))
    println(results[1, :])
    plot(0.0:stepsize:1.0, transpose(results))
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
