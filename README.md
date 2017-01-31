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

Sources of the data sets (I have preprocessed some of them):

- Facebook: J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS, 2012; via [SNAP](http://snap.stanford.edu/data/egonets-Facebook.html)
- US Powergrid: Watts, D. J., Strogatz, S. H., 1998. Collective dynamics of “small-world” networks. Nature 393, 440-442; via [Tore Opsahl](https://toreopsahl.com/datasets/#uspowergrid)
- Euroroad: [KONECT](http://konect.uni-koblenz.de/networks/subelj_euroroad)
- Philadelphia transportation: [TransportationNetworks](https://github.com/bstabler/TransportationNetworks/tree/a6a968de8f6db1bc15ff0ff3b19ebd8a50afc79e/Philadelphia)
- Austin transportation: [TransportationNetworks](https://github.com/bstabler/TransportationNetworks/tree/a6a968de8f6db1bc15ff0ff3b19ebd8a50afc79e/Austin)


