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

    # wherever meaningful, there is a normalized (*_n) and an unnormalized variant
    const information_functions = Dict(
        :degree => g -> degree(g),
        :degree_n => g -> degree(g) / maxdegrees,
        :inv_degree => g -> 1 ./ (degree(g) + eps()),
        :inv_degree_n => g -> (1 ./ (degree(g) + eps())) / maxdegrees,
        :local_clustering => g -> local_clustering_coefficient(g),
        :neighborhood2 => g -> map(v -> length(neighborhood(g, v, 2)), vertices(g)),
        :neighborhood2_n => g -> map(v -> length(neighborhood(g, v, 2)), vertices(g)) / maxdegrees,
        :neighborhood3 => g -> map(v -> length(neighborhood(g, v, 3)), vertices(g)),
        :neighborhood3_n => g -> map(v -> length(neighborhood(g, v, 3)), vertices(g)) / maxdegrees
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
function smoothcurves{T<:Number}(curves::AbstractArray{T, 2}, use_loess = true)
    average_curve = squeeze(mapslices(mean, curves, 1), 1)

    if use_loess
        stepsize = 1 / (size(curves)[end] - 1)
        range = 0.0:stepsize:1.0
        return predict(loess(range, average_curve), range)
    else
        return average_curve
    end
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





####################################################################################################
# TRAINING AND TESTING
####################################################################################################

function log_annealing(;kwargs...)
    args = Dict(kwargs)
    # push!(error_trace, args[:f_new])
    
    if args[:steps] % 100 == 0
        println(args[:steps], " steps, T = ", args[:T])
        println("\tParameters: ", args[:x_new])
        println("\tError: ", args[:f_new])
    end
end


function trainandtest(name::String, train_graph::Graph, test_graph::Graph,
                      results_file::String, variant::String;
                      mode = "a", parameters...)
    for (s, v) in parameters
        @eval $s = $v
    end

    println("Training on ", name, " using ", variant, "...")
    trained_param = annealing(percolation_loss(train_graph, infos, samples, stepsize),
                              fill(0.0, length(infos)),
                              init_temp, temp_decay, exploration_size,
                              (s, v) -> s > iterations;
                              debug_callback = log_annealing)
    println("Learned parameters (", name, "): ", trained_param, "\n")

    baseline_dist = uninformed_probabilities(test_graph)
    optimized_dist = informed_probabilities(test_graph, trained_param, infos)

    baseline_samples = samplepercolations(max_component_size, test_graph,
                                          baseline_dist, samples, stepsize)
    optimized_samples = samplepercolations(max_component_size, test_graph,
                                           optimized_dist, samples, stepsize)

    baseline_curve = normalizecurves(baseline_samples)
    optimized_curve = normalizecurves(optimized_samples)

    range = 0.0:stepsize:1.0
    open(results_file, mode) do f
        for (x, y, y2) in zip(range,
                              smoothcurves(baseline_curve, false),
                              smoothcurves(baseline_curve, true))
            write(f, "$name $variant baseline $x $y $y2\n")
        end
        
        for (x, y, y2) in zip(range,
                          smoothcurves(optimized_curve, false),
                          smoothcurves(optimized_curve, true))
            write(f, "$name $variant optimized $x $y $y2\n")
        end
    end
end


function fixtest(name::String, test_graph::Graph, trained_param::AbstractArray{Float64, 1},
              results_file::String, variant::String; mode::String = "a", parameters...)
    for (s, v) in parameters
        @eval $s = $v
    end

    baseline_dist = uninformed_probabilities(test_graph)
    optimized_dist = informed_probabilities(test_graph, trained_param, infos)

    baseline_samples = samplepercolations(max_component_size, test_graph,
                                          baseline_dist, samples, stepsize)
    optimized_samples = samplepercolations(max_component_size, test_graph,
                                           optimized_dist, samples, stepsize)

    baseline_curve = normalizecurves(baseline_samples)
    optimized_curve = normalizecurves(optimized_samples)

    range = 0.0:stepsize:1.0
    open(results_file, mode) do f
        for (x, y, y2) in zip(range,
                              smoothcurves(baseline_curve, false),
                              smoothcurves(baseline_curve, true))
            write(f, "$name $variant baseline $x $y $y2\n")
        end
        for (x, y, y2) in zip(range,
                          smoothcurves(optimized_curve, false),
                          smoothcurves(optimized_curve, true))
            write(f, "$name $variant optimized $x $y $y2\n")
        end
    end
end


function main()
    fb1 = readnetwork("../data/facebook1.txt")
    fb2 = readnetwork("../data/facebook2.txt")
    austin = readnetwork("../data/austin.txt")
    philadelphia = readnetwork("../data/philadelphia.txt")
    random1 = Graph(2000, 5000)
    random2 = Graph(2000, 5000)
    
    examples = ["facebook" => (fb1, fb2),
                "traffic" => (austin, philadelphia),
                "random" => (random1, random2)]

    
    info_variants = Dict("d" => [:degree],
                         "i" => [:inv_degree],
                         "l" => [:local_clustering],
                         "n" => [:neighborhood2],
                         "dl" => [:degree, :local_clustering],
                         "dln" => [:degree, :local_clustering, :neighborhood2])

    info_variants_n = Dict("i" => [:inv_degree_n],
                           "d" => [:degree_n],
                           "l" => [:local_clustering],
                           "n" => [:neighborhood2_n],
                           "dl" => [:degree_n, :local_clustering],
                           "dln" => [:degree_n, :local_clustering, :neighborhood2_n])
    
    training_parameters = Dict(
        :stepsize => 0.05,
        :samples => 10,
        :init_temp => 10.0,
        :temp_decay => 0.9,
        :exploration_size => 0.05,
        :iterations => 400)

    # for (variant, infos) in info_variants
    #     for (name, (train, test)) in examples
    #         trainandtest(name, train, test,
    #                      "../evaluation/results1.txt", variant;
    #                      mode = "a", infos = infos, training_parameters...)
    #     end
    # end

    # for (variant, infos) in info_variants_n
    #     for (name, (train, test)) in examples
    #         trainandtest(name, train, test,
    #                      "../evaluation/results2.txt", variant;
    #                      mode = "a", infos = infos, training_parameters...)
    #     end
    # end

    p1_train = [("facebook", fb1, "d") => [0.5],
                ("traffic", austin, "d") => [0.65],
                ("random", random1, "d") => [0.25],
                ("facebook", fb1, "i") => [-1.1],
                ("traffic", austin, "i") => [0.25],
                ("random", random1, "i") => [0.4],
                ("facebook", fb1, "l") => [1.1],
                ("traffic", austin, "l") => [0.2],
                ("random", random1, "l") => [0.2],
                ("facebook", fb1, "n") => [0.05],
                ("traffic", austin, "n") => [0.5],
                ("random", random1, "n") => [0.05],
                ("facebook", fb1, "dl") => [0.400757,0.737475],
                ("traffic", austin, "dl") => [0.188876,-0.214628],
                ("random", random1, "dl") => [0.301184,-0.765181],
                ("facebook", fb1, "dln") => [0.185994,-0.475402,0.0138872],
                ("traffic", austin, "dln") => [0.194423,0.255597,0.73907],
                ("random", random1, "dln") => [0.147955,-0.133006,0.0134066]]

    for ((name, graph, variant), params) in p1_train
        fixtest(name, graph, params,
                "../evaluation/training1.txt", variant;
                mode = "a", infos = info_variants[variant], training_parameters...)
    end
    
end



end
