# Network Science - Project 2#

## Motivation: destroying networks fast ##

My goal is to learn a way of informed percolation which optimally disconnects a network.  This can be useful, e.g., for 
preventing desease spreading by optimally targeting persons to be vaccinated, or to find out which nodes in a network 
should be prioritized in protection (or conversely, attacked with priority...).

## Method ##

For that purpose, I want to try taking into account some local node information `info` (like degree, local clustering 
coefficient, triangles, etc.), and learn the optimal way to combine it. This amounts to having a function

    f(node) = σ(wᵀ · info(node)),
    
(where `σ` is the logistic function) which we use as the distribution for percolation.  We then want to minimize the 
expected area under the percolation curve with respect to the weights `w`, estimated by simulating a few percolation 
runs using it.

## Experimental setup ##

I use simulated annealing to optimize the problem, because it requires fewer calls to the target function, which in 
this case is quite costly.  Furthermore, the sample runs simulated in the loss function can be run in parallel.

To compare different network types, I choose several ones from different domains (roads, social, infrastructure, random),
and compare the result obtained by optimization to the baseline of uniform percolation.

# LICENSE #

All code and text by me are [unlicensed](http://unlicense.org/).

The data sets are taken from [Konect](http://konect.uni-koblenz.de), and thus subject to the 
[Creative Commons Attribution-ShareAlike 2.0 Germany License](http://konect.uni-koblenz.de/license).
