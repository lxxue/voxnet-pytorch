import numpy as np

class Jitter(object):
    def __init__(self, max_jitter_ij, max_jitter_k):
        self.max_ij = max_jitter_ij
        self.max_k = max_jitter_k

    
    def __call__(self, batch):
        # batch x D x D x D
        # or batch x nview x D x D xD
        dst = src.copy()
        rank = len(batch.shape)
        # reverse in x/y direction
        shitf_ijk = [np.random.randint(-self.max_ij, self.max_ij),
                     np.random.randint(-self.max_ij, self.max_ij),
                     np.random.randint(-self.max_k, self.max_k)]
        if rank == 4:
            if np.random.binomial(1, .2):
                dst[:, ::-1, :, :] = dst
            if np.random.binomial(1, .2):
                dst[:, :, ::-1, :] = dst
            for axis, shift in enumerate(shift_ijk):
                if shift != 0:
                    dst = np.roll(dst, shift, axis+1)
        elif rank == 5:
            if np.random.binomial(1, .2):
                dst[:, :, ::-1, :, :] = dst
            if np.random.binomial(1, .2):
                dst[:, :, :, ::-1, :] = dst
            for axis, shift in enumerate(shift_ijk):
                if shift != 0:
                    dst = np.roll(dst, shift, axis+2)
        else:
            assert "rank of each batch should be 4 or 5 instead of {}".format(rank)

        return dst

