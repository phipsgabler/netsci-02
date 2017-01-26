module NetSci02

using LightGraphs, GraphPlot
using Distributions

include("utils.jl")

export readnetwork


function optimize(loss::Function, initial_parameter, temp_decay, stepsize, stop)
    const k = length(initial_parameter)

    step = 1
    x_old = initial_parameter
    f_old = loss(initial_parameter)
    T = 1.0

    while !stop(step, f_old)
        direction = normalize(randn(k))
        x_new = x_old + direction * stepsize
        f_new = loss(x_new)
        scaled_diff = (f_old - f_new) / T

        if rand() <= min(1, exp(scaled_diff))
            x_old = x_new
            f_old = f_new
        end

        T = max(temp_decay * T, eps())
        step += 1
    end

    return x_old
end



function samplepercolations(measure::Function, graph::Graph, repetitions::Integer,
                            result_type = typeof(measure(Graph(1, 0))))
    results = Array(result_type, repetitions, nv(graph))
    samplepercolations!(results, graph, repetitions, measure)
    return results
end

function samplepercolations!{T}(results::AbstractArray{T, 2}, graph::Graph, repetitions::Integer,
                                measure::Function)
    const vertices = nv(graph)
    for r = 1:repetitions
        victim = copy(graph)
        for v = vertices:-1:1
            results[r, v] = measure(victim)
            rem_vertex!(victim, rand(1:v))
        end
    end
end



function test(;nv = 100, ne = 50)
    graph = Graph(nv, ne)
    samplepercolations(graph, 10) do g
        length(connected_components(g)), length(triangles(g))
    end
end

end
