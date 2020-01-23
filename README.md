### PyABS ###

Code to simulate purchase of portfolio of Asset Backed Securities under simulated interest rate scenarios. This code is intended to illustrate opportunities presented by a real life scenario, the Term Asset Loan Facility, explained in tis article: https://towardsdatascience.com/alpha-generation-using-data-science-quantitative-analysis-abs-talf-part-1-eade08b075c

![Image of concept](https://cdn-images-1.medium.com/max/800/1*6DcNyLMw3rwAw1JOvsgUMg.png)

### Steps ###

*   Generate correlated random numbers to feed a multivariate process applied to interest rates
*   Simulate movements of spreads over a becnhmark for given asset classes
*   Simulate purchases of assets under given assumptions
*   Simulate probabilities of assets transitioning from intial rating to other ratings and defaults, using a Markov process
*   Calculate distribution of returns/risks under several scenarios

### Contact: ###
* Luis M Sanchez:	<lmsanch@gmail.com>
