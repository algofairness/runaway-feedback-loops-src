import numpy.random
import random

##############################################################################

class PolyaUrn:
    """A basic Polya Urn with a given update matrix"""

    def __init__(self, initial_state, update_matrix):
        self.initial_state = initial_state
        self.update_matrix = update_matrix
        self.reset()

    def reset(self):
        self.state = self.initial_state

    def after_draw(self, choice, state):
        return state

    def update_row(self, choice, state):
        return self.update_matrix[choice]

    def draw(self, draws=1):
        rr = random.random
        binom = numpy.random.binomial
        ss = self.state
        ad = self.after_draw
        ur = self.update_row
        choices = { False: 1, True: 0 }
        for i in range(draws):
            v = rr() * (ss[0] + ss[1])
            choice = choices[v<ss[0]]
            update_row = ur(choice, ss)
            ss = (ss[0] + update_row[0], ss[1] + update_row[1])
            ss = ad(choice, ss)
            yield ss
        self.state = ss

##############################################################################

def add_truncation(cls, truncation):
    class TruncatedUrn(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
        def update_row(self, choice, state):
            ur = cls.update_row(self, choice, state)
            return (min(ur[0], truncation),
                    min(ur[1], truncation))
    return TruncatedUrn

def add_poisson_update(cls):
    poisson = numpy.random.poisson
    class PoissonUrn(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
        def update_row(self, choice, state):
            ur = cls.update_row(self, choice, state)
            return poisson(ur)
    return PoissonUrn

def add_exponential_decay(cls, prob):
    binom = numpy.random.binomial
    ad = cls.after_draw
    class ExponentiallyDecayingUrn(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
        def after_draw(self, choice, state):
            state = ad(self, choice, state)
            decay = binom(state, prob)
            state = (max(state[0] - decay[0], 1),
                     max(state[1] - decay[1], 1))
            return state
    return ExponentiallyDecayingUrn

def add_linear_surprise(cls):
    rr = random.random
    class GoodPolyaUrn(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
        def update_row(self, choice, ss):
            r = rr()
            t = sum(ss)
            v = ((t - ss[0]) / t,
                 (t - ss[1]) / t)
            mx = max(v[0], v[1])
            v = (v[0] / mx, v[1] / mx)
            if r >= v[choice]:
                return (0,0)
            return cls.update_row(self, choice, ss)
    return GoodPolyaUrn

def add_partial_surprise(cls, param):
    rr = random.random
    class GoodPolyaUrn(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
        def update_row(self, choice, ss):
            r = rr()
            t = sum(ss)
            v = ((t - ss[0]) / t,
                 (t - ss[1]) / t)
            mx = max(v[0], v[1])
            v = (v[0] / mx, v[1] / mx)
            if r >= v[choice]:
                return param
            return cls.update_row(self, choice, ss)
    return GoodPolyaUrn

def add_weighted_surprise(cls, param):
    rr = random.random
    class GoodPolyaUrn(cls):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
        def update_row(self, choice, ss):
            r = rr()
            t = sum(ss)
            v = ((t - ss[0]) / t,
                 (t - ss[1]) / t)
            mx = max(v[0], v[1])
            v = (v[0] / mx, v[1] / mx)
            if r >= v[choice]:
                return param[0:2]
            da = self.update_matrix[0][0] - param[0]
            db = self.update_matrix[1][1] - param[1]
            ra = param[0]
            rb = param[1]
            wd = param[2]
            wr = param[3]
            if choice == 0:
                 return (da + ra, rb / wr)
            return (ra / wr, db + rb)
    return GoodPolyaUrn

def add_sqrt_surprise(cls):
    rr = random.random
    class BetterPolyaUrn(cls):
        """FIXME: I don't remember what this does. I remember
        that we worked out the family of all possible compensations
        and that we believed this converged faster. But I don't remember
        the details.

        We're not going to use this for now, but let's not forget it exists.
        """
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)

        def update_row(self, choice, ss):
            r = rr()
            t = sum(ss)
            ps = (ss[0] / t, ss[1] / t)
            v = ((ps[1] / ps[0]) ** 0.5,
                 (ps[0] / ps[1]) ** 0.5)
            mx = max(v[0], v[1])
            v = (v[0] / mx, v[1] / mx)
            if r >= v[choice]:
                return (0,0)
            return cls.update_row(self, choice, ss)
    return BetterPolyaUrn

##############################################################################

class BatchedPoissonUrn:

    def __init__(self, initial_state, update_parameters, truncation):
        self.initial_state = initial_state
        self.update_parameters = update_parameters
        self.truncation = truncation
        self.reset()

    def reset(self):
        self.state = self.initial_state

    def draw(self, draws=1):
        rr = random.random
        ss = self.state
        poisson = numpy.random.poisson
        binom = numpy.random.binomial
        lamb = self.update_parameters
        truncation = self.truncation

        for i in range(draws):
            v = ss[0] / (ss[0] + ss[1])
            t = binom(truncation, v)

            cops_0 = t
            cops_1 = truncation - t
            crimes_0 = poisson(lamb[0])
            crimes_1 = poisson(lamb[1])

            arrests_0 = min(cops_0, crimes_0)
            arrests_1 = min(cops_1, crimes_1)

            updates_0 = 0
            updates_1 = 0

            r = rr()
            if cops_0 > 0 and r > ss[0] / ((ss[0] + ss[1]) * cops_0):
                updates_0 = arrests_0
            r = rr()
            if cops_1 > 0 and r > ss[1] / ((ss[0] + ss[1]) * cops_1):
                updates_1 = arrests_1
            ss = (ss[0] + updates_0, ss[1] + updates_1)
            yield ss
        self.state = ss



class MixedUrn:

    def __init__(self, initial_state, d_a, d_b, r_a, r_b):
        self.update_matrix = ((d_a + r_a,       r_b),
                              (      r_a, d_b + r_b))
        self.initial_state = initial_state
        self.state = initial_state

    def reset(self):
        self.state = self.initial_state

    def draw(self, draws=1):
        rr = random.random
        ss = self.state
        um = self.update_matrix
        for i in range(draws):
            v = rr() * (ss[0] + ss[1])
            if v < ss[0]:
                update_row = um[0]
            else:
                update_row = um[1]
            ss = (ss[0] + update_row[0], ss[1] + update_row[1])
            yield ss
        self.state = ss
