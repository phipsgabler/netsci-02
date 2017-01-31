module NetSci02

using LightGraphs
using Distributions
using Loess

include("utils.jl")
include("tests.jl")


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
                            stepsize = 0.05,
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
        :degree => g -> degree(g) / maxdegrees,
        :inv_degree => g -> 1 ./ (degree(g) / maxdegrees + eps()),
        :local_clustering => g -> local_clustering_coefficient(g),
        :neighborhood2 => g -> map(v -> length(neighborhood(g, v, 2)), vertices(g)) / maxdegrees,
        :neighborhood3 => g -> map(v -> length(neighborhood(g, v, 3)), vertices(g)) / maxdegrees
    )
    
    information = Array(Float64, nv(graph), length(infos))
    for (i, info) in enumerate(infos)
        information[:, i] = information_functions[info](graph)
    end

    return makedistributions(information * weights)
end


"""
    uninformed_probabilities(graph::Graph)

Like `informed_probabilities`, but just use the same distribution based on `φ` for every node.
"""
function uninformed_probabilities(graph::Graph)
    return φ -> Bernoulli.(fill(φ, nv(graph)))
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
    max_component_size(g::Graph)

Calculate the size of the larges connected component of `g`.
"""
max_component_size(g::Graph) = reduce(max, 0, length.(connected_components(g)))


"""
    percolation_loss(graph::Graph, infos::Vector{Symbol}, samples, stepsize)

Construct a loss function which samples the expected area of the percolation curve of `graph`,
using the parameter as weightings for local information.
"""
function percolation_loss(graph::Graph, infos::Vector{Symbol}, samples = 10, stepsize = 0.05,
                          measure = max_component_size)
    function loss(parameter)
        dist = informed_probabilities(graph, parameter, infos)
        curves = samplepercolations(measure, graph, dist, samples, stepsize)
        return percolation_area(curves)
    end
end


function main()
    fb1 = readnetwork("../data/facebook1.txt")
    fb2 = readnetwork("../data/facebook2.txt")
    austin = readnetwork("../data/austin.txt")
    philadelphia = readnetwork("../data/philadelphia.txt")

    infos = [:degree, :local_clustering, :neighborhood2]
    stepsize = 0.05
    samples = 10

    # train stuff on fb1 and austin
    init_temp, temp_decay, exploration_size = 10.0, 0.9, 0.05
    iterations = 400

    function log_annealing(;kwargs...)
        args = Dict(kwargs)
        # push!(error_trace, args[:f_new])
        
        if args[:steps] % 100 == 0
            println(args[:steps], " steps, T = ", args[:T])
            println("\tParameters: ", args[:x_new])
            println("\tError: ", args[:f_new])
        end
    end
    
    train(graph) = annealing(percolation_loss(graph, infos, samples, stepsize),
                             fill(0.0, length(infos)),
                             init_temp, temp_decay, exploration_size,
                             (s, v) -> s > iterations;
                             debug_callback = log_annealing)

    fb1_param = train(fb1)
    austin_param = train(austin)
    println("Learned parameters (fb1): ", fb1_param)
    println("Learned parameters (austin): ", austin_param)


    # validate stuff on fb2 and philadelphia
    fb2_dists_baseline = uninformed_probabilities(fb2)
    philadelphia_dists_baseline = uninformed_probabilities(philadelphia)
    fb2_dists_optimized = informed_probabilities(fb2, fb1_param, infos)
    philadelphia_dists_optimized = informed_probabilities(philadelphia, austin_param, infos)

    fb2_samples_baseline = samplepercolations(max_component_size, fb2, fb2_dists_baseline,
                                              samples, stepsize)
    philadelphia_samples_baseline = samplepercolations(max_component_size, philadelphia,
                                                       philadelphia_dists_baseline,
                                                       samples, stepsize)
    fb2_samples_optimized = samplepercolations(max_component_size, fb2, fb2_dists_optimized,
                                               samples, stepsize)
    philadelphia_samples_optimized = samplepercolations(max_component_size, philadelphia,
                                                        philadelphia_dists_optimized,
                                                        samples, stepsize)

    fb2_curve_baseline = smoothcurves(normalizecurves(fb2_samples_baseline))
    philadelphia_curve_baseline = smoothcurves(normalizecurves(philadelphia_samples_baseline))
    fb2_curve_optimized = smoothcurves(normalizecurves(fb2_samples_optimized))
    philadelphia_curve_optimized = smoothcurves(normalizecurves(philadelphia_samples_optimized))

    range = 0.0:stepsize:1.0
    open("../evaluation/results.txt", "w") do f
        for (x, y) in zip(range, fb2_curve_baseline)
            write(f, "fb2 baseline $x $y \n")
        end
        for (x, y) in zip(range, fb2_curve_optimized)
            write(f, "fb2 optimized $x $y \n")
        end
        for (x, y) in zip(range, philadelphia_curve_baseline)
            write(f, "philadelphia baseline $x $y \n")
        end
        for (x, y) in zip(range, philadelphia_curve_optimized)
            write(f, "philadelphia optimized $x $y \n")
        end
    end
end



end
