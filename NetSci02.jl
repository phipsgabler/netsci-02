module NetSci02

using LightGraphs, GraphPlot
using Distributions

include("utils.jl")


"""
    annealing(loss::Function, initial_parameter, temp_decay, stepsize, stop)

Find a parameter setting minimizing `loss`, using simulated annealing.
"""
function annealing(loss::Function, initial_parameter, initial_temperature, temp_decay,
                   stepsize, stop)
    const k = length(initial_parameter)

    steps = 1
    T = initial_temperature
    x_old = initial_parameter
    f_old = loss(initial_parameter)

    while !stop(steps, f_old)
        direction = normalize(randn(k)) # uniform on hypersphere
        x_new = x_old + direction * stepsize
        f_new = loss(x_new)

        if rand() <= min(1, exp(-(f_new - f_old) / T))
            x_old = x_new
            f_old = f_new
        end

        T = max(temp_decay * T, eps())
        steps += 1

        if steps % 10 == 0
            println("$steps steps, T = $T ")
            println("\t", x_new)
            println("\t", f_new)
        end
    end

    return x_old
end


"""
    samplepercolations(measure::Function, graph::Graph, repetitions::Integer,
                       getdistribution::Function)

Sample `repetitions` percolation runs on `graph`, calculating `measure` each time.  The
probability of deleting a nodes is calculated by `getdistribution` of the current graph.
"""
function samplepercolations(measure::Function,
                            graph::Graph,
                            repetitions::Integer,
                            getdistribution::Function,
                            result_type = typeof(measure(Graph(1, 0))))
    const vertices = nv(graph)
    results = SharedArray(result_type, repetitions, nv(graph))
    pmap(1:repetitions) do r
        victim = copy(graph)
        for v = vertices:-1:1
            results[r, v] = measure(victim)
            rem_vertex!(victim, rand(getdistribution(victim)))
        end
    end
    return results
end


"""
    informed_probabilities(graph::Graph, weights::Vector{Float64})

Construct a distribution over nodes which integrates knowledge about local clustering,
degree, and second-order neighbourhood, taking into account `weights`.
"""
function informed_probabilities(graph::Graph, weights::Vector{Float64})::DiscreteUnivariateDistribution
    const maxdegrees = nv(graph) * (nv(graph) + 1) / 2
    infos = Array(Float64, nv(graph), 3)
    infos[:, 1] = local_clustering_coefficient(graph)
    infos[:, 2] = degree(graph) / maxdegrees
    infos[:, 3] = map(v -> length(neighborhood(graph, v, 2)), vertices(graph)) / maxdegrees

    Categorical(softmax(infos * weights))
end


"""
    percolation_loss(graph, samples)

Construct a loss function which samples the expected area of the percolation curve of `graph`,
using the parameter as weightings.
"""
function percolation_loss(graph::Graph, samples::Integer)
    function loss(parameter)
        dist(g) = informed_probabilities(g, parameter)
        curves = samplepercolations(graph, samples, dist) do g
            maximum(length(c) for c in connected_components(g))
        end
        normalizers = mapslices(maximum, curves, 2)
        # every curve is normed relative to its maximum (highest) value; then, average
        sum(curves ./ normalizers) / samples
    end
end


function test(;nv = 100, ne = 50)
    graph = Graph(nv, ne)
    # samplepercolations(graph, 10, _ -> DiscreteUniform(1, nv)) do g
    #     length(connected_components(g)), length(triangles(g))
    # end

    annealing(percolation_loss(graph, 10), rand(3), 10, 0.99, 0.01, (s, v) -> s > 1000)
end

end
