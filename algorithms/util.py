from math import log
import tensorflow as tf
import numpy as np

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
        sum += prob1 * log(prob1 / prob2, 2)
    return sum


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def flatten_grad(grads, var_list):
    return tf.concat([tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(var_list, grads)], 0)


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    # in numpy
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
