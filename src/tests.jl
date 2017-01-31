using PyPlot

function test_sampling(graph = Graph(1000, 1000))
    const stepsize = 0.05
    const range = 0.0:stepsize:1.0
    
    dist1 = makedistributions(-2 ./ degree(graph) + 2 * degree(graph))
    dist2 = φ -> fill(Bernoulli(φ), nv(graph))

    # println(map(φ -> params(dist1(φ)[1]), range))
    # println(10 ./ degree(graph))
    
    results1 = samplepercolations(graph, dist1, 10, stepsize) do g
        reduce(max, 0, length.(connected_components(g)))
    end

    println(percolation_area(results1))
    # println(results1[1, :])
    plot(0.0:stepsize:1.0, smoothcurves(normalizecurves(results1)), color = "r")

    results2 = samplepercolations(graph, dist2, 10, stepsize) do g
        reduce(max, 0, length.(connected_components(g)))
    end

    println(percolation_area(results2))
    # println(results2[1, :])
    plot(0.0:stepsize:1.0, smoothcurves(normalizecurves(results2)), color = "g")
end


function test_annealing(graph = Graph(1000, 1000))
    error_trace = Float64[]
    infos = [:inv_degree, :degree]

    function log_annealing(;kwargs...)
        args = Dict(kwargs)
        push!(error_trace, args[:f_new])
        
        if args[:steps] % 10 == 0
            println(args[:steps], " steps, T = ", args[:T])
            println("\tParameters: ", args[:x_new])
            println("\tError: ", args[:f_new])

            if args[:updated]
                dist = informed_probabilities(graph, args[:x_new], infos)
                curves = samplepercolations(graph, dist, 1, 0.05) do g
                    reduce(max, 0, length.(connected_components(g)))
                end
                plot(0.0:0.05:1.0, smoothcurves(normalizecurves(curves)))
            end
        end
    end

    params = annealing(percolation_loss(graph, infos, 10), fill(0.0, length(infos)),
                       10.0, 0.9, 0.05, (s, v) -> s > 500;
                       debug_callback = log_annealing)

    figure()
    plot(error_trace)
end
