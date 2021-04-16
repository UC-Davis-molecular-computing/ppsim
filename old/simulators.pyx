# ''' to enable line tracing
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# '''
import cython
import numpy as np
cimport numpy as np
import time
from IPython.display import clear_output
import datetime
import random
import itertools

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64
from numpy.random.c_distributions cimport (random_hypergeometric, random_interval)
from libc.math cimport log, lgamma, sqrt


cpdef np.int64_t sample_coll(np.int64_t n, np.int64_t r, double logn, double u):
    if r >= n:
        return 1

    # samples the random variable giving the number of draws from the n agents before the first collision
    cdef np.int64_t t_lo, t_mid, t_hi
    cdef logu, lhs

    logu = log(u)
    lhs = lgamma(n-r+1) - logu

    t_lo = int(sqrt(n - r) / 2)  # starting guess ~ expected value
    t_hi = min(t_lo * 8, n-r)

    # otherwise, we maintain the invariant that l > t_lo and l <= t_hi
    # and do binary search until returning t_lo + 1 = t_hi
    while t_lo > 0 and lhs < lgamma(n-r+1-t_lo) + t_lo * logn:
        t_lo = t_lo // 2   # shrink t_lo until t_lo < l
    if t_lo == 0: # in this case 1 >= l, so return l = 1
        return 1
    while t_hi < n - r and lhs >= lgamma(n-r+1-t_hi) + t_hi * logn:
        t_hi = min(t_hi * 2, n-r)  # grow t_hi until l <= t_hi

    while t_lo < t_hi - 1:
        t_mid = (t_lo + t_hi) // 2
        if lhs < lgamma(n-r+1-t_mid) + t_mid * logn:
            t_hi = t_mid
        else:
            t_lo = t_mid
    return t_hi


cpdef int linear_search(np.int64_t[::1] counts, int [::1] order, np.int64_t x):
    # counts is a list of counts, x is integer <= sum(counts)
    # finds index of the entry in counts belonging to x by linear search
    # order is an array giving the sorting of counts which we will use for linear search
    cdef int i=0
    while x >= 0:
        x -= counts[order[i]]
        i += 1
    return order[i - 1]


cpdef int sample_agent(np.int64_t [::1] config, int [::1] order, np.int64_t x):
    # modifies config to remove the sampled agent, and returns the index sampled from
    # if size == 0:
    #     size = np.sum(config)
    cdef int index = linear_search(config, order, x)
    config[index] -= 1
    return index



cdef multivariate_hypergeometric(np.ndarray[np.int64_t, ndim=1] colors, np.int64_t nsamples, np.ndarray[np.int32_t, ndim=1] order, bitgen_t * bitgen_state):
    cdef size_t l = colors.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] output = np.zeros(l, dtype=np.int64)
    cdef Py_ssize_t i = 0
    cdef np.int64_t total = np.sum(colors)
    while nsamples > 0 and i < l - 1:
        total -= colors[order[i]]
        output[order[i]] = random_hypergeometric(bitgen_state, colors[order[i]], total, nsamples)
        nsamples -= output[order[i]]
        i += 1
    output[order[i]] = nsamples
    return output


def time_update(t, t_max, start_time, current_time):
    new_current_time = time.perf_counter()
    if new_current_time - current_time > 1: # give update every 1 second
        current_time = new_current_time
        # clear_output(wait = True)
        print(f'{t / t_max * 100:.4}% complete')
        est_remaining_seconds = int((current_time - start_time) * (t_max / t - 1))
        print(f'Estimated time remaining: ' + str(datetime.timedelta(seconds=est_remaining_seconds)))
        return new_current_time
    return current_time


def simulate_sequential_array(delta, np.int64_t [::1] dist, np.int64_t t_max, np.int64_t snapshot_threshold = 1):
    # delta is the transition rule, a dictionary mapping tuples of states
    # dist is a list of counts of states
    cdef np.int64_t n = sum(dist) # number of agents
    cdef int q = len(dist) # number of states
    cdef np.ndarray[np.int64_t, ndim = 1] C = np.array(list(itertools.chain(*[[i] * dist[i] for i in range(len(dist))])), dtype = np.int64)
    history = []
    ts = []
    cdef np.int64_t t = 0
    cdef np.int64_t a, b
    start_time = current_time = time.perf_counter()
    while t < t_max:
        if t % snapshot_threshold == 0:
            ts.append(t)
            history.append(np.array([np.count_nonzero(C == i) for i in range(q)]))
            if t > 0:
                current_time = time_update(t, t_max, start_time, current_time)

        # get a random pair (a,b)
        a = b = random.uniform(0, n-1)
        while a == b:
            b = random.uniform(0, n-1)

        # update states at (a,b)
        C[a], C[b] = delta[(C[a], C[b])]
        t += 1
    return np.array(ts) / n, np.array(history)


def simulate_multi_batch(dict delta, np.int64_t [::1] dist, np.int64_t t_max, np.int64_t snapshot_threshold = 1, seed = None, rng = np.random.default_rng()):
    # delta is the transition rule, a dictionary mapping tuples of states
    # dist is a list of counts of states
    cdef np.int64_t n = sum(dist) # number of agents
    cdef double logn = np.log(n)
    cdef int q = len(dist) # number of states
    cdef np.int64_t sort_threshold = max(int(np.log(q)), 2) # how many iterations to re-sort C
    cdef np.int64_t batch_threshold = int(np.sqrt(n)) # how many interactions per batch, will be dynamically optimized
    cdef bitgen_t *bitgen
    gen = PCG64(seed=seed)
    capsule = gen.capsule
    bitgen = <bitgen_t *> PyCapsule_GetPointer(capsule, "BitGenerator")

    ts = [0]
    cdef np.int64_t t = 0
    cdef np.ndarray[long, ndim=3] delta_array = np.zeros((q,q,q), dtype=int)

    cdef int i, j, a, b, c
    symmetric_transitions = set()
    for i in range(q):
        for j in range(q):
            a, b = delta[(i,j)]
            delta_array[(a,i,j)] += 1
            delta_array[(b,i,j)] += 1
            # this reaction is symmetric
            if (b,a) == delta[(j,i)]:
                symmetric_transitions.add((i,j))

    # picks sender / receiver for a, b if the reaction is asymmetric
    def delta_on_unordered_pair(int a, int b):
        if (a,b) in symmetric_transitions:
            return delta[(a,b)]
        else:
            if random.getrandbits(1):
                return delta[(a,b)]
            else:
                new_b,new_a = delta[(b,a)]
                return new_a, new_b

    cdef np.ndarray[np.int64_t, ndim=2] D = np.zeros((q,q),dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] C = np.array(dist,dtype=np.int64) # current configuration
    history = [C]
    cdef np.ndarray[np.int32_t, ndim=1] order = np.array(np.argsort(-C), dtype = np.int32)
    cdef np.ndarray[np.int64_t, ndim=1] updated_counts
    cdef np.int64_t num_delayed, num_updated, l
    cdef double t1, t2, t3, start_time, current_time, u, r
    cdef np.ndarray[np.int64_t, ndim=1] row_sums, row

    start_time = current_time = time.perf_counter()

    while t < t_max:
        if len(ts) % sort_threshold == 0:
            order = np.array(np.argsort(-C), dtype = np.int32)

        updated_counts = np.zeros(q,dtype=np.int64)
        # start with count 2 of delayed agents (the next interaction)
        num_delayed = 2
        num_updated = 0

        t1 = time.perf_counter()

        while num_delayed + num_updated < batch_threshold:
            u = bitgen.next_double(bitgen.state)
            l = sample_coll(n, num_delayed + num_updated, logn, u=u)
            num_delayed += 2 * ((l-1) // 2)
            # add (l-1) // 2 pairs of delayed agents, the lth agent a was already picked, so has a collision
            # sample if a was a delayed or an updated agent
            u = bitgen.next_double(bitgen.state)
            r = num_delayed / (num_delayed + num_updated)
            if u <= r:
                # if a was delayed, need to first update a to current its current state
                # c is the delayed partner that a interacted with, so add this interaction
                x = random_interval(bitgen, n - num_updated - 1)
                a = sample_agent(C, order, random_interval(bitgen, n - 1 - num_updated))
                c = sample_agent(C, order,random_interval(bitgen, n - 1 - num_updated - 1))
                a, c = delta_on_unordered_pair(a,c)
                t += 1
                # c is moved from delayed to updated, a is currently uncounted
                updated_counts[c] += 1
                num_updated += 1
                num_delayed -= 2
            else:
                # if a was updated, we simply sample a and remove it from updated counts
                a = sample_agent(updated_counts, order, random_interval(bitgen, num_updated - 1))
                num_updated -= 1

            if l % 2 == 0:  # when l is even, the collision must with with a formally untouched agent
                b = sample_agent(C, order, random_interval(bitgen, n - 1- num_updated - 1))
            else: # when l is odd, the collision is with the next agent, either untouched, delayed, or updated
                u = bitgen.next_double(bitgen.state)
                r = num_updated / (n-1)
                if u < r:
                    # b is an updated agent, simply remove it
                    b = sample_agent(updated_counts, order, random_interval(bitgen, num_updated - 1))
                    num_updated -= 1
                else:
                    # we simply remove b from C is b is untouched
                    b = sample_agent(C, order, random_interval(bitgen, n - 1- num_updated - 1))
                    # if b was delayed, we have to do the past interaction
                    r = (num_updated + num_delayed) / (n - 1)
                    if u < r:
                        c = sample_agent(C, order, random_interval(bitgen, n - 1- num_updated - 2))
                        b, c = delta_on_unordered_pair(b,c)
                        t += 1
                        updated_counts[c] += 1
                        num_updated += 1
                        num_delayed -= 2

            a, b = delta_on_unordered_pair(a,b)
            t += 1
            updated_counts[a] += 1
            updated_counts[b] += 1
            num_updated += 2

        t2 = time.perf_counter()
        row_sums = multivariate_hypergeometric(C, num_delayed // 2, order, bitgen)
        C = C - row_sums

        for i in range(q):
            row = multivariate_hypergeometric(C, row_sums[i], order, bitgen)
            D[i] = row
            C = C - row

        updated_counts += np.sum(delta_array * D, axis = (1,2))
        t += num_delayed // 2
        C += updated_counts

        if t - ts[-1] >= snapshot_threshold:
            ts.append(t)
            history.append(C.copy())

        t3 = time.perf_counter()

        # dynamically update batch threshold
        batch_threshold = int(((t3 - t2) / (t2 - t1)) ** 0.1 * batch_threshold)

        current_time = time_update(t, t_max, start_time, current_time)
    # clear_output(wait = True)
    print(f'Finished!')
    return np.array(ts) / n, np.array(history)