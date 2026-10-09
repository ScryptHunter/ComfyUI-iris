"""Positional seeding: every stochastic draw is a pure function of position.

Training randomness (timesteps, noise, dropout masks, caption choice) is keyed
on (seed, rank, epoch, position) instead of an ambient RNG stream, so a resumed
run reproduces the unbroken run bit-exactly from any checkpoint.
"""


def mix_seed(*parts: int) -> int:
    """Mix integers into one 63-bit seed (splitmix64 finalizer per part)."""
    mask = 0xFFFFFFFFFFFFFFFF
    h = 0
    for p in parts:
        h = (h + (p & mask) + 0x9E3779B97F4A7C15) & mask
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & mask
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & mask
        h ^= h >> 31
    return h & 0x7FFFFFFFFFFFFFFF

