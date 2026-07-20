"""Softmax-temperature (tau) anneal for the soft-Voronoi objective.

"tau is the whole ballgame" (README):
    * too large  -> mushy membership, capacity under-enforced, points clump.
    * too small  -> near-hard assignment, vanishing gradients, training stalls.

Strategy: start WARM (large tau) for a stable global signal, then decay toward HARD so the
final soft membership matches the scipy Voronoi partition the validators score. Log the
soft-vs-hard capacity gap every eval; if it diverges, tau is too large.

tau is a distance^2 temperature; points live in [0,1], so a sensible tau_start is on the
order of the squared mean spacing (~1/N) and tau_end an order of magnitude smaller.
"""

import math


class TauSchedule:
    def __init__(self, tau_start=0.02, tau_end=0.002, warmup=0, total=1, mode="cosine"):
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.warmup = int(warmup)
        self.total = max(1, int(total))
        self.mode = str(mode)

    def value(self, step):
        if step < self.warmup:
            return self.tau_start
        progress = (step - self.warmup) / max(1, self.total - self.warmup)
        progress = min(1.0, max(0.0, progress))
        if self.mode == "linear":
            f = 1.0 - progress
        elif self.mode == "exp":
            # geometric interpolation in log-space
            log = math.log(self.tau_start) + progress * (math.log(self.tau_end) - math.log(self.tau_start))
            return math.exp(log)
        else:  # cosine
            f = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.tau_end + (self.tau_start - self.tau_end) * f
