import numpy
from scipy import linalg, stats
import math
from matplotlib import pyplot


class poisson_gen(stats.rv_discrete):
    def _pmf(self, k, mu):
        return math.exp(-mu) * mu**k / math.gamma(k+1)


class norm_gen(stats.rv_discrete):
    def _pmf(self, k, mu):
        return math.exp(-1/2 * (k-mu)**2) / math.sqrt(2*math.pi)


def normal_exec(N):
    mu = 5
    norm = norm_gen(name="normal")
    x = numpy.linspace(0, 15, num=N)

    v = []
    for e in x:
        v.append(norm._pmf(e, mu=mu))
    fig, ax = pyplot.subplots(1, 1)
    ax.plot(x, v)

    r = stats.norm.rvs(size=N)+mu
    ax.hist(r, bins=10, density=True)

    ax2 = ax.twinx()
    prob = stats.poisson.cdf(x, mu=5)
    ax2.plot(x, prob)

    ax2.set_ylabel("CDF")
    ax.set_ylabel("PMF")
    pyplot.show(block=False)
    return r


def poisson_exec(N):
    mu = 5
    poisson = poisson_gen(name="poisson")
    x = numpy.linspace(0, 20, num=N)

    v = []
    for e in x:
        v.append(poisson._pmf(e, mu))
    fig, ax = pyplot.subplots(1, 1)
    ax.plot(x, v)

    r = stats.poisson.rvs(mu, size=N)
    ax.hist(r, bins=10, density=True)

    ax2 = ax.twinx()
    prob = stats.poisson.cdf(x, mu)
    ax2.plot(x, prob)

    ax2.set_ylabel("CDF")
    ax.set_ylabel("PMF")
    pyplot.show(block=False)
    return r


if __name__ == '__main__':
    print("LINEAR ALGEBRA")
    # A is singular. Can't solve with b. Find numerical inverse of A instead.
    A = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = numpy.array([1, 2, 3])
    A_inv = linalg.pinv(A)
    x = numpy.dot(A_inv, b)
    print(numpy.dot(A, x))

    B = numpy.random.rand(3, 3)
    x = numpy.dot(A_inv, b)
    print(numpy.dot(A, x))

    evals, evecs = linalg.eig(A)
    print(evals)
    print(evecs)

    A_inv = linalg.pinv(A)
    print(linalg.det(A_inv))

    print(linalg.norm(A, ord=1))
    print(linalg.norm(A, ord=2))

    print("STATISTICS")
    N=1000
    data_poisson = poisson_exec(N)
    data_normal = normal_exec(N)

    check = stats.ttest_ind(data_poisson, data_normal)
    null_hypotheis = "Null hypothesis: The two samples have the same average."
    print(null_hypotheis)
    print(f"p-value: {check.pvalue}")
    if check.pvalue < 0.01:
        print("Reject null hypothesis")
    else:
        print("Cannot reject null hypothesis")

        
