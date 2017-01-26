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


