library(dplyr)
library(ggplot2)

## read.table('results_unnormalized.txt', col.names = c('network', 'phase', 'p', 'value')) %>%
pdf('results.pdf', 10, 7)
read.table('results1.txt', col.names = c('sample', 'variant', 'phase', 'p', 'value', 'value_smooth')) %>%
    filter(variant != 'i') %>%
    ggplot(aes(p, value_smooth, color = phase)) +
    facet_grid(variant~sample) +
    geom_line() +
    labs(x = 'Expected remaining nodes', y = 'Relative size of largest cluster') +
    theme(legend.position = 'bottom')
dev.off()

read.table('results2.txt', col.names = c('sample', 'variant', 'phase', 'p', 'value', 'value_smooth')) %>%
    ## filter(phase == 'optimized') %>%
    ggplot(aes(p, value, color = phase)) +
    facet_grid(variant~sample) +
    geom_line() +
    labs(x = 'Expected remaining nodes', y = 'Relative size of largest cluster')

pdf('training.pdf', 10, 7)
read.table('training1.txt', col.names = c('sample', 'variant', 'phase', 'p', 'value', 'value_smooth')) %>%
    filter(variant != 'i') %>%
    ggplot(aes(p, value_smooth, color = phase)) +
    facet_grid(variant~sample) +
    geom_line() +
    labs(x = 'Expected remaining nodes', y = 'Relative size of largest cluster') +
    theme(legend.position = 'bottom')
dev.off()
