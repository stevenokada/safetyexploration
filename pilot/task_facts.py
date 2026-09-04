"""
Real-world multi-hop fact composition, up to 5 hops.

Why: three lens runs on the synthetic tasks found nothing, most likely because
their intermediates are single DIGITS -- flagged by the workspace paper as poorly
lens-aligned. Published work that DID decode intermediates (Brauer et al., 80-95%)
used word-valued intermediates. This puts intermediates back in the vocabulary.

Chain (each step a real, memorised fact, none given in context):
  currency -> country -> capital -> first letter -> element -> atomic number
  e.g.  lev -> Bulgaria -> Sofia -> S -> Sulfur -> 16

  k=1  capital of a named country
  k=2  + currency to country
  k=3  + first letter of the capital
  k=4  + element whose symbol is that letter
  k=5  + that element's atomic number

Design notes
  * Currencies are restricted to those belonging to EXACTLY ONE country, or
    "the country whose currency is the X" has several correct answers.
  * Single-token-ness is RECORDED PER HOP rather than required. Requiring every
    hop to be single-token left one usable fact, because element names are mostly
    multi-token. The lens can read whichever hops are single tokens; the rest
    still serve the behavioural depth measurement.
  * k>=4 needs the capital's first letter to be a single-letter element symbol,
    so the pool shrinks with depth. Available pool size per k is reported below.
"""
import random

ELEM = {"B":("Boron",5), "C":("Carbon",6), "F":("Fluorine",9), "H":("Hydrogen",1),
        "I":("Iodine",53), "K":("Potassium",19), "N":("Nitrogen",7), "O":("Oxygen",8),
        "P":("Phosphorus",15), "S":("Sulfur",16), "U":("Uranium",92),
        "V":("Vanadium",23), "W":("Tungsten",74), "Y":("Yttrium",39)}

# (currency, country, capital) -- currency unique to that one country
FACTS = [
    ("yen","Japan","Tokyo"), ("baht","Thailand","Bangkok"), ("zloty","Poland","Warsaw"),
    ("forint","Hungary","Budapest"), ("shekel","Israel","Jerusalem"),
    ("rupiah","Indonesia","Jakarta"), ("sol","Peru","Lima"), ("lev","Bulgaria","Sofia"),
    ("ruble","Russia","Moscow"), ("krona","Sweden","Stockholm"),
    ("yuan","China","Beijing"), ("koruna","Czechia","Prague"), ("leu","Romania","Bucharest"),
    ("dram","Armenia","Yerevan"), ("manat","Azerbaijan","Baku"),
    ("bolivar","Venezuela","Caracas"), ("afghani","Afghanistan","Kabul"),
    ("hryvnia","Ukraine","Kyiv"), ("tenge","Kazakhstan","Astana"),
    ("naira","Nigeria","Abuja"), ("cedi","Ghana","Accra"), ("kyat","Myanmar","Yangon"),
    ("dong","Vietnam","Hanoi"), ("taka","Bangladesh","Dhaka"), ("kip","Laos","Vientiane"),
    ("won","South Korea","Seoul"), ("rand","South Africa","Pretoria"),
    ("lari","Georgia","Tbilisi"), ("birr","Ethiopia","Addis Ababa"),
    ("riel","Cambodia","Phnom Penh"),
]

def pool(k):
    """Facts that can support a chain of this depth."""
    if k <= 3:
        return FACTS
    return [f for f in FACTS if f[2][0] in ELEM]

def generate(rng, k):
    """Returns (question, gold, intermediates, reasoning)."""
    if not 1 <= k <= 5:
        raise ValueError("k must be 1..5")
    p = pool(k)
    if not p:
        raise RuntimeError(f"no facts support k={k}")
    cur, country, capital = rng.choice(p)
    L = capital[0]
    if k == 1:
        q = f"What is the capital city of {country}?"
        chain = [capital]
        r = f"The capital of {country} is {capital}."
        return q, capital, chain, r
    base = f"the country whose currency is the {cur}"
    r2 = f"The {cur} is the currency of {country}. The capital of {country} is {capital}."
    if k == 2:
        return f"What is the capital city of {base}?", capital, [country, capital], r2
    if k == 3:
        return (f"What is the first letter of the capital city of {base}?",
                L, [country, capital, L], r2 + f" Its first letter is {L}.")
    el, num = ELEM[L]
    r3 = r2 + f" Its first letter is {L}, and {L} is the symbol for {el}."
    if k == 4:
        return (f"Which chemical element has the symbol that is the first letter "
                f"of the capital city of {base}?", el, [country, capital, L, el], r3)
    return (f"What is the atomic number of the chemical element whose symbol is the "
            f"first letter of the capital city of {base}?",
            str(num), [country, capital, L, el, str(num)],
            r3 + f" The atomic number of {el} is {num}.")

if __name__ == "__main__":
    for k in range(1, 6):
        print(f"pool at k={k}: {len(pool(k))} facts")
    print()
    for k in range(1, 6):
        q, g, c, r = generate(random.Random(4), k)
        print(f"k={k}: {q}\n   gold={g!r}  chain={c}\n")
