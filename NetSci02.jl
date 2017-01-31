module NetSci02

using LightGraphs, GraphPlot
using Distributions
using Loess
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


"""
    informed_probabilities(graph::Graph, weights::Vector{Float64}, infos::Vector{Symbol})

Construct a paramtrized vector of distributions over nodes which integrates knowledge about local
clustering, degree, and second-order neighbourhood, taking into account `weights`.  `infos` can
be used to choose which information to include.
"""
function informed_probabilities(graph::Graph,
                                weights::Vector{Float64},
                                infos::Vector{Symbol})::Function
    const maxdegrees = nv(graph) * (nv(graph) + 1) / 2
    const information_functions = Dict(
        :degree => g -> degree(g),
        :inv_degree => g -> 1 ./ (degree(g) + eps()),
        :local_clustering => g -> local_clustering_coefficient(g),
        :neighbourhood2 => g -> map(v -> length(neighborhood(g, v, 2)), vertices(g)) / maxdegrees,
        :neighbourhood3 => g -> map(v -> length(neighborhood(g, v, 3)), vertices(g)) / maxdegrees
    )
    
    information = Array(Float64, nv(graph), length(infos))
    for (i, info) in enumerate(infos)
        information[:, i] = information_functions[info](graph)
    end

    return makedistributions(information * weights)
end


"""
    makedistributions(logits::Vector{Float64})::Function

Convert a vector of weight values for every node (`logits`) into a parametrized Bernoulli
distribution for each node, such that the expected fraction of nodes after using this for
percolation depends on the parameter (but the individual probabilities may differ).
"""
function makedistributions(logits::AbstractArray{Float64, 1})::Function
    probabilities = logistic.(logits)
    return function (φ)
        scaled_probabilities = probabilities .* (φ / mean(probabilities))
        Bernoulli.(clamp(scaled_probabilities, 0.0, 1.0)) # because rounding errors
    end
end


"""
    smoothcurves(curves::AbstractArray{Float64, 2})

Return the smoothed average of the given `curves` (using Loess regression).
"""
function smoothcurves{T<:Number}(curves::AbstractArray{T, 2})
    stepsize = 1 / (size(curves)[end] - 1)
    range = 0.0:stepsize:1.0
    average_curve = squeeze(mapslices(mean, curves, 1), 1)
    loess_curve = predict(loess(range, average_curve), range)
end


"""
    normalizecurves(curves::AbstractArray{Float64, 2})

Normalize all curves to the range [0, 1] by their maximum.
"""
normalizecurves{T<:Number}(curves::AbstractArray{T, 2}) = curves ./ mapslices(maximum, curves, 2)


"""
    percolation_area{T<:Number}(curves::AbstractArray{T, 2}, smooth = true)

Estimate the area under the sampled percolation curves by averaging and calculating the area.
If `smooth` is true, uses Loess regression and numerical integration, otherwise simply sum up and
weight by the stepsize.
"""
function percolation_area{T<:Number}(curves::AbstractArray{T, 2}, smooth = true)
    normalized_curves = normalizecurves(curves)
    samples = size(curves)[1]
    stepsize = 1 / (size(curves)[end] - 1)

    if smooth
        # numerical integration over a local regression curve
        average_curve = squeeze(mapslices(mean, normalized_curves, 1), 1)
        smoothed_curve = loess(0.0:stepsize:1.0, average_curve)
        return quadgk(x -> predict(smoothed_curve, x), 0.0, 1.0)[1]
    else
        # just take the average of everything...
        return sum(normalized_curves) * stepsize / samples
    end
end


"""
    percolation_loss(graph::Graph, infos::Vector{Symbol}, samples, stepsize)

Construct a loss function which samples the expected area of the percolation curve of `graph`,
using the parameter as weightings for local information.
"""
function percolation_loss(graph::Graph, infos::Vector{Symbol}, samples = 10, stepsize = 0.05)
    function loss(parameter)
        dist = informed_probabilities(graph, parameter, infos)
        curves = samplepercolations(graph, dist, samples, stepsize) do g
            reduce(max, 0, length.(connected_components(g)))
        end

        return percolation_area(curves)
    end
end






function simulate_random(graph = Graph(1000, 1000))
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


function test(graph = Graph(1000, 1000))
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

end
