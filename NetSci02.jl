module NetSci02

using LightGraphs, GraphPlot
using Distributions

export readnetwork

const line_regex = r"^(\d+)\s(\d+)"


"""
    readnetwork(filename::String, limit::Number = Inf; fromzero::Bool = false)

Read a space separated file into an (undirected) Graph in an efficient way.

# Arguments
* `limit::Number`: Maximum number of lines to read (good for large files)
* `fromzero::Bool`: Whether vertices are counted from zero; if so, correct accordingly
"""
function readnetwork(filename::String, limit::Number = Inf; fromzero::Bool = false)
    graph = Graph()
    vertices = 0
    const correction = Int(fromzero)

    # using grep to exclude comment lines
    open(filename) do file
        for (l, line) in enumerate(eachline(file))
            if l > limit
                break
            end

            if (m = match(line_regex, line)) !== nothing
                raw_v1, raw_v2 = m.captures
                v1 = parse(Int, raw_v1) + correction
                v2 = parse(Int, raw_v2) + correction

                new_vertices = max(v1, v2)
                if new_vertices > vertices
                    @assert add_vertices!(graph, new_vertices - vertices)
                    vertices = new_vertices
                end
                
                # we explicitely ignore inverse directions, if there are any
                @assert has_edge(graph, v1, v2) || add_edge!(graph, v1, v2)
            end
        end
    end

    return graph
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
