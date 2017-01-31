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

logistic(x) = 1 / (1 + exp(-x))
softmax(xs) = let e = exp(xs); e / sum(e) end


"""
    annealing(loss::Function, initial_parameter, initial_temperature, temp_decay, stepsize, stop)

Find a parameter setting minimizing `loss`, using simulated annealing.
"""
function annealing(loss::Function, initial_parameter, initial_temperature,
                   temp_decay, stepsize, stop;
                   debug_callback = (;kwargs...) -> return)
    const k = length(initial_parameter)

    steps = 1
    T = initial_temperature
    x_old = initial_parameter
    f_old = loss(initial_parameter)

    while !stop(steps, f_old)
        direction = normalize(randn(k)) # uniform on hypersphere
        x_new = x_old + direction * stepsize
        f_new = loss(x_new)
        
        if (updated = rand() <= min(1, exp(-(f_new - f_old) / T)))
            x_old = x_new
            f_old = f_new
        end

        debug_callback(steps = steps, T = T, updated = updated, direction = direction,
                       x_old = x_old, f_old = f_old,
                       f_new = f_new, x_new = x_new)
        
        T = max(temp_decay * T, eps())
        steps += 1
    end

    return x_old
end

