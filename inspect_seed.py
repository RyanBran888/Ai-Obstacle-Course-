from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

from Architecture.coop_env.rng import SeededRandom, MAX_SEED


def parse_master(value: str | None) -> int | str | None:
    if value is None or value.lower() == "none":
        return None
    # integer-ish?
    try:
        return int(value)
    except Exception:
        return value


def main(argv: list[str] | None = None) -> None:
    p = ArgumentParser(description="Inspect episode seeds from the master SeededRandom stream")
    p.add_argument("--master", type=str, default=None, help="master seed (int or string); omit or 'None' for system-random)")
    p.add_argument("--episode", type=int, default=3000, help="episode index (how many draws to advance; default 3000)")
    p.add_argument("--show-all", action="store_true", help="print every drawn seed instead of only when width changes")
    args = p.parse_args(argv)

    master = parse_master(args.master)
    rng = SeededRandom(master, label="episodes")
    upper = MAX_SEED - 1  # randint upper bound: 0 .. 2**63-2

    seed = None
    prev_bitlen: int | None = None
    prev_hexlen: int | None = None
    for i in range(1, args.episode + 1):
        seed = rng.randint(0, upper)
        bitlen = seed.bit_length()
        hexlen = len(hex(seed)) - 2  # omit '0x'

        if args.show_all:
            print(f"#{i}: {seed}  {hex(seed)}  bit_length={bitlen}  hex_len={hexlen}")
        else:
            changed = False
            if prev_bitlen is None or bitlen != prev_bitlen:
                print(f"#{i}: bit_length -> {bitlen}  (seed={seed})")
                changed = True
            if prev_hexlen is None or hexlen != prev_hexlen:
                # hex length usually changes in lockstep with bitlen, but report anyway
                if not changed:
                    print(f"#{i}: hex_len -> {hexlen}  (seed={seed})")

        prev_bitlen = bitlen
        prev_hexlen = hexlen

    if seed is None:
        print("no seed drawn")
        return


if __name__ == "__main__":
    main()
