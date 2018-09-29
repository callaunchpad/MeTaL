from math import log

def discreteKLDivergence(dist1, dist2):
	"""
	Calculates the discrete time KL divergence between two PDFs

	@Author Hank O'Brien
	>>> discreteKLDivergence([1,2,3,4],[1,2,3,4])
	0.0
	>>> float('%.5f' % discreteKLDivergence([1,2,3,5],[1,2,3,4]))
	1.60964
	"""

	sum = 0
	for prob1, prob2 in zip(dist1, dist2):
		sum += prob1 * log(prob1/prob2, 2)
	return sum

if __name__ == "__main__":
    import doctest
    doctest.testmod()