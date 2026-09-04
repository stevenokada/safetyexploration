"""
Real-world multi-hop fact composition, up to 6 hops, plus a matched parallel control.

Reformulation that unlocks depth
--------------------------------
The first version asked "what is the capital of the country whose currency is X",
which caps out at three hops because the answer type changes as you extend it
(city, then letter, then element, then number) and the currency->country relation
has nowhere further back to go.

Fixing the ANSWER and extending backwards removes both limits. The answer is
always the atomic number; k controls how far back the question starts:

  k=1  atomic number of Vanadium
  k=2  ... of the element whose symbol is V
  k=3  ... whose symbol is the first letter of Vienna
  k=4  ... the first letter of the capital of Austria
  k=5  ... the capital of the country where Mozart was born
  k=6  ... where the composer of The Magic Flute was born

Every depth answers with a number, so one answer-format instruction serves all of
them, and each added hop is a genuine memorised lookup rather than a rephrasing.

The parallel control
--------------------
Same fact pool, same kind of retrieval, k of them, but independent: given k
countries, name the one whose capital comes first alphabetically. Work is k
lookups; serial depth is 2 (retrieve, then compare) however large k gets.
"""
import random

# work, relation word, creator, country, capital, letter, element, atomic number
#
# The relation word matters: an earlier version asked for "the composer of The
# Metamorphosis" (a novella) and "the composer of the Ninth Symphony" (Beethoven,
# Mahler, Dvorak, Bruckner and Schubert all wrote one). Chain-of-thought accuracy
# at the deepest level was 0.575, which is a malformed-question rate, not a depth
# measurement. Works are now paired with the right relation, and works with more
# than one famous claimant are removed.
CHAINS = [
    ("The Magic Flute",    "composer", "Mozart",    "Austria", "Vienna",     "V", "Vanadium",   23),
    ("The Metamorphosis",  "author",   "Kafka",     "Czechia", "Prague",     "P", "Phosphorus", 15),
    ("A Doll's House",     "author",   "Ibsen",     "Norway",  "Oslo",       "O", "Oxygen",      8),
    ("The Snow Queen",     "author",   "Andersen",  "Denmark", "Copenhagen", "C", "Carbon",      6),
    ("Finlandia",          "composer", "Sibelius",  "Finland", "Helsinki",   "H", "Hydrogen",    1),
    ("the Minute Waltz",   "composer", "Chopin",    "Poland",  "Warsaw",     "W", "Tungsten",   74),
    ("The Cherry Orchard", "author",   "Chekhov",   "Russia",  "Moscow",     "M", None,          0),
]
CHAINS = [c for c in CHAINS if c[6]]   # drop any whose letter is not an element symbol

# country -> capital, for the parallel control (a wider pool: no element constraint)
CAPITALS = [
    ("Austria","Vienna"), ("Czechia","Prague"), ("Norway","Oslo"), ("Denmark","Copenhagen"),
    ("Hungary","Budapest"), ("Finland","Helsinki"), ("Germany","Berlin"), ("Poland","Warsaw"),
    ("Japan","Tokyo"), ("Thailand","Bangkok"), ("Israel","Jerusalem"), ("Indonesia","Jakarta"),
    ("Peru","Lima"), ("Bulgaria","Sofia"), ("Russia","Moscow"), ("Sweden","Stockholm"),
    ("China","Beijing"), ("Egypt","Cairo"), ("Kenya","Nairobi"), ("Cuba","Havana"),
    ("Portugal","Lisbon"), ("Greece","Athens"), ("Ireland","Dublin"), ("Morocco","Rabat"),
]

MAX_K = 6

def generate(rng, k):
    """Serial chain. Returns (question, gold, intermediates, reasoning)."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be 1..{MAX_K}")
    work, relation, creator, country, capital, letter, element, num = rng.choice(CHAINS)
    stem = {
        1: f"the element {element}",
        2: f"the element whose symbol is {letter}",
        3: f"the element whose symbol is the first letter of {capital}",
        4: f"the element whose symbol is the first letter of the capital city of {country}",
        5: (f"the element whose symbol is the first letter of the capital city of "
            f"the country where {creator} was born"),
        6: (f"the element whose symbol is the first letter of the capital city of "
            f"the country where the {relation} of {work} was born"),
    }[k]
    q = f"What is the atomic number of {stem}?"
    # Intermediates are what the model must resolve that the prompt does NOT name.
    # At depth k the prompt names full[5-k], so everything after it is un-emitted.
    full = [creator, country, capital, letter, element]
    chain = full[6 - k:]
    steps = []
    if k >= 6: steps.append(f"the {relation} of {work} is {creator}")
    if k >= 5: steps.append(f"{creator} was born in {country}")
    if k >= 4: steps.append(f"the capital of {country} is {capital}")
    if k >= 3: steps.append(f"the first letter of {capital} is {letter}")
    if k >= 2: steps.append(f"{letter} is the symbol for {element}")
    steps.append(f"the atomic number of {element} is {num}")
    return q, str(num), chain + [str(num)], "; ".join(steps).capitalize() + "."

def generate_parallel(rng, k, agg="unique_letter"):
    """agg="unique_letter" (default): exactly one capital starts with letter L --
    which country? A single-character test per item, no ordering to maintain.

    agg="alphabetical": which capital comes first alphabetically. This was the
    first design and it FAILED as a control -- the model sat at or below chance
    from k=2 to k=12 even though chain-of-thought solved it every time, because a
    k-way character-level comparison is itself beyond latent reach. Kept so the
    two aggregations can be compared: same retrievals, different combining cost.
    """
    if agg == "unique_letter":
        for _ in range(400):
            picks = rng.sample(CAPITALS, k)
            first = [cap[0] for _, cap in picks]
            uniq = [i for i, L in enumerate(first) if first.count(L) == 1]
            if not uniq:
                continue
            i = rng.choice(uniq)
            country, capital = picks[i]
            names = ", ".join(c for c, _ in picks)
            q = (f"Consider these {k} countries: {names}.\n"
                 f"Exactly one of them has a capital city beginning with the letter "
                 f"{capital[0]}. Which country is it? Answer with just the country name.")
            reasoning = ("; ".join(f"{c}'s capital is {cap}" for c, cap in picks) +
                         f". Only {capital} begins with {capital[0]}, so the answer is {country}.")
            return q, country, [cap for _, cap in picks], reasoning
        raise RuntimeError(f"could not build a unique-letter instance at k={k}")
    return _generate_parallel_alpha(rng, k)


def _generate_parallel_alpha(rng, k):
    """Matched parallel control: k independent capital lookups, then one comparison.
    Serial depth 2 regardless of k. Returns (question, gold, intermediates, reasoning)."""
    if k < 2:
        raise ValueError("parallel control needs k >= 2")
    picks = rng.sample(CAPITALS, k)
    winner = min(picks, key=lambda p: p[1])
    names = ", ".join(c for c, _ in picks)
    q = (f"Consider these {k} countries: {names}.\n"
         f"Which one has the capital city whose name comes first alphabetically? "
         f"Answer with just the country name.")
    reasoning = ("; ".join(f"{c}'s capital is {cap}" for c, cap in picks) +
                 f". Alphabetically first is {winner[1]}, so the answer is {winner[0]}.")
    return q, winner[0], [cap for _, cap in picks], reasoning

if __name__ == "__main__":
    for k in range(1, MAX_K + 1):
        q, g, c, r = generate(random.Random(2), k)
        print(f"k={k}: {q}\n   gold={g}  intermediates={c}")
    print()
    q, g, c, r = generate_parallel(random.Random(1), 5)
    print(f"parallel k=5: {q}\n   gold={g}  capitals={c}")
