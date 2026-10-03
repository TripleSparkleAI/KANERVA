# KANERVA_TERMS - the words SETTLE uses for sparse distributed memory

SETTLE names every part of a sparse distributed memory with Kanerva's own word. Where Kanerva's word is
two words, SETTLE joins them with a hyphen, so a keyword reads as one thing: `read-address:`,
`address-noise:`, `hard-locations:`. This page records each word, the sentence Kanerva wrote it in, and
where that sentence is.

## Sources

The quotes below come from these texts. Page numbers are the printed numbers.

- **K1993**: P. Kanerva, "Sparse Distributed Memory and Related Models", in M. H. Hassoun (ed.),
  *Associative Neural Memories*, Oxford University Press 1993, pp. 50-76. On disk:
  `wikis/WIKI_SDR/papers/sdm/kanerva-1993-sdm-and-related-models.pdf`. Its pages are numbered P. 1 to
  P. 41 in the author's 2002 typescript, and those are the numbers cited here.
- **FKB1989**: M. J. Flynn, P. Kanerva, N. Bhadkamkar, *Sparse Distributed Memory: Principles and
  Operation*, Stanford CSL-TR-89-400, December 1989 (Kanerva is a co-author). Fetched from
  `http://i.stanford.edu/pub/cstr/reports/csl/tr/89/400/CSL-TR-89-400.pdf`.
- **K2009**: P. Kanerva, "Hyperdimensional Computing", *Cognitive Computation* 1:139-159, 2009. On disk:
  `wikis/WIKI_SDR/papers/kanerva-2009-hyperdimensional-computing.pdf`.
- **K2010**: P. Kanerva, "What We Mean When We Say 'What's the Dollar of Mexico?'", AAAI Fall Symposium
  FS-10-08, 2010. On disk: `wikis/WIKI_SDR/papers/kanerva-2010-dollar-of-mexico.pdf`.
- **K1988**: P. Kanerva, *Sparse Distributed Memory*, MIT Press 1988. The book is not on disk and no
  free text of it was found. Its quotes here come through a source that quotes it with a page number,
  and that source is named. Google Books lists "access circle", "hard locations", "critical distance"
  and "Best-Match Machine" among the book's common terms.

## The table

| SETTLE keyword or term | Kanerva's own word | Kanerva's sentence | Where | Plain meaning |
|---|---|---|---|---|
| `read-address:` | retrieval address; retrieval cue; memory cue | "When x is used as the retrieval address, the locations activated by x are pooled" | K1993 P. 6 | The address you read at. SETTLE says read because the statement is `s.read`. |
| | | "The retrieval cue for the sequence can be noisy" | K1993 P. 2 | |
| | | "The address A is also called a memory cue." | K2010 p. 2 | |
| `write-address` (prose; `s.write :cat` names it) | storage address; write address | "x Storage or retrieval address; contents of the address register" | K1993 P. 10 | The address a word is stored at. |
| | | "a word can be read back not only by giving the original write address but also by giving one close to it" | FKB1989 p. 1 | |
| `address-noise:` | noise; a noisy address | "Nine noisy words (20% noise) are stored, and the tenth is used as a retrieval cue." | K1993 P. 2, Fig. 3.1 | The fraction of the read-address's bits flipped at random before the read. `address-noise: 0.2` flips a fifth of them. |
| | | "If the memory is probed with a noisy address A', the retrieved pattern X' will usually have some noise" | K2009 p. 144 | |
| `hard-locations:` | hard locations | "the hard locations are so few compared to the number of possible addresses" | K1993 P. 5 | The memory locations that are built, each with a fixed random address. |
| `activation-radius:` | activation radius; radius of activation | "the binomial distribution ... can be used to find the activation radius H that corresponds to a given probability p of activating a location" | K1993 P. 5 | The Hamming distance within which a hard location is activated by an address. |
| | | "H Radius of activation (e.g., H = 447 bits)" | K1993 P. 10 | |
| `activation-probability:` | probability of activation, p | "p Probability of activation (e.g., p = 0.000445 ...). This important parameter determines the number of hard locations that are activated, on the average, by an address" | K1993 P. 9-10 | The fraction of hard locations one address activates. SETTLE derives the activation radius from it. |
| `access-circle` (prose) | access circle; the set activated by x | "Address space, hard locations, and the set activated by x. H is the (Hamming) radius of activation." | K1993 Fig. 3.5 | The hard locations within the activation radius of an address. K1988 calls the set the access circle. |
| `data-word` (prose) | data word; input word; output word | "Except for the lengths of the address and data words, the memory resembles ordinary computer memory." | K1993 P. 3 | The word stored and read back. In SETTLE's autoassociative memories the data word is the pattern itself. |
| `word-size:` | word size | "The capacity of a location is referred to as the memory's word size, U" | K1993 P. 3 | The number of bits in an address and in a data word. SETTLE uses one length for both. |
| `bit-counters` (prose) | counters; up-down counters | "C Contents matrix; U x M up-down counters with range c" | K1993 P. 10 | The integer counters each hard location keeps, one per bit of the data word. A write adds +1 or -1 to each. |
| `read-threshold` (prose) | threshold value 0 | "the sums are compared to a threshold value 0 to get an output vector z" | K1993 P. 6 | A read sums the bit-counters of the activated hard locations and keeps each bit whose sum is positive. |
| `iterated-reads:` | iterations; iterated reading | "Such recovery typically takes several iterations (fewer than ten) where the address X' is used to retrieve X''" | K2009 p. 144 | The most times a read uses its own result as the next read-address. The read stops early when the result stops changing. |
| | | "now Z(1) can be used as a new retrieval cue ... This iterative process ... produces a sequence of outputs ... that converges rapidly" | FKB1989 p. 15 | |
| `critical-distance` (prose) | critical distance | "the distance beyond which divergence is more likely than convergence" | K1988 p. 63, quoted by Linhares et al., *Front. Hum. Neurosci.* 2014, doi:10.3389/fnhum.2014.00222 | How far a read-address can be from a stored word and still converge to it under iterated reads. |
| | | "if [the cue] is closer than a certain critical distance to P(1), then summing and thresholding ... results in a vector Z(1) that is closer to P(1)" | FKB1989 p. 15 | |
| `best-match` (prose) | the best-match problem | "An apparently simple way in which this best-match problem could be tackled is the following" | FKB1989 p. 7 | Finding the stored word nearest a given word. K1988's term index lists the "Best-Match Machine". |
| `tolerate-noise:` | (SETTLE's own; built from Kanerva's "noise") | none | | The address-noise a memory is sized to tolerate. SETTLE picks the activation radius that recalls best at that noise. |

## The words SETTLE retired

| Retired keyword | New keyword | Why it went |
|---|---|---|
| `cue:` | `read-address:` | Kanerva says "retrieval cue", but he says "retrieval address" in the same chapter, and the address names the thing being read at. `read-address` pairs with the statement `s.read`. |
| `damage:` | `address-noise:` | Kanerva never says damage. He says noise, and he says which thing is noisy: the address. |
| `locations:` | `hard-locations:` | Kanerva's term. A plain "location" is any point of the address space; a hard location is one that is built. |
| `radius:` | `activation-radius:` | Kanerva's term, K1993 P. 5. Our wiki also says "access radius"; Kanerva's chapter says "activation radius" and "radius of activation", so SETTLE takes his. |
| `fire:` | `activation-probability:` | Kanerva's p, "probability of activation". A location is activated in his words; it does not fire. |
| `iterations:` | `iterated-reads:` | Names what is iterated. |
| `size:` (on `sdm`, `sdmscale`, `softsdm`, `contenttrack`, `refusal`) | `word-size:` | Kanerva's term, K1993 P. 3. The Hopfield `memory` statement keeps `size:`, because it is not Kanerva's memory. |
| `tolerate:` | `tolerate-noise:` | Names what is tolerated. |

A program that uses a retired keyword stops with an error that names the new one, for example
`line 3: cue: is now read-address: (Kanerva's retrieval address)`.

## Keywords that stay

`key:` (SETTLE's keyed text), `via:` (which read), `wake:`, `fade:`, `seed:`, `rounds:`, `samples:`,
`mode:`, `burn:`, `softness:`, `gain:`, `write_samples:`, `load:` and `level:` are SETTLE's own and
name no Kanerva concept.
