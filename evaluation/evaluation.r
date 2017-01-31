library(dplyr)
library(ggplot2)

read.table('results.txt', col.names = c('network', 'phase', 'p', 'value')) %>%
    ggplot(aes(p, value, color = phase)) +
    facet_wrap(~network) +
    geom_line()
