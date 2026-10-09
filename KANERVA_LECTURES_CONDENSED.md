# KANERVA_LECTURES_CONDENSED - what an SDM is, in Kanerva's own words from his recorded lectures

A condensed version of Pentti Kanerva's recorded lectures on sparse distributed memory, made for THE EXPLAINER at
the top of the site's SDM EXPLORE page (`SETTLE/settle-site/src/sdmexplore/Explainer.jsx`) and for the simpler intros on
the other SDM pages. It keeps his framing and his terms. Every quotation is a line of YouTube's auto-caption,
verbatim and unpunctuated, as the captioner heard him, with the lecture it comes from and the time it was said. A
quotation is never paraphrased inside quotation marks; the paraphrase around it is ours.

## The sources

The transcripts live in the research vault `_research/kanerva-lectures/` (gitignored; its `INDEX.md` is committed and
lists every talk, its channel, date, length and provenance). Each lecture is cited by its YouTube id; the URL is
`https://www.youtube.com/watch?v=<id>&t=<seconds>s`.

| id | what it is | used here for |
|---|---|---|
| WOcYfaXEAFY | the sparse distributed memory lecture at the Redwood Center, UC Berkeley: the memory built step by step from 0:37 to 0:53, then mapped onto the cerebellum and onto a computer's RAM (published on the curation channel under the title "Computing in Superposition in the Classical Domain"; the title is swapped with hH65_yWSpFc, see the vault index) | the memory itself: addresses, hard locations, writing, reading, noise, the fixed addresses, RAM, the cerebellum |
| GNKbGmXYY0I | "Neural Computation Lecture" (hyperdimensional computing, with the associative memory as its last part) | the best match and the memory's "I don't know" |
| NJW2NacQpzM | "The Brain's Circuits Suggest Computing with High Dimensional Vectors" | a noisy address still retrieves |
| zUCoxhExe0o | Stanford Seminar, "Computing with High-Dimensional Vectors", 2017 | the circuit of the cerebellum, the lifetime of learning |
| 1g5VEcnG6fI | "The computer and the brain" | the item memory, the nearest neighbour |
| hH65_yWSpFc | a rehearsal of a DARPA talk on computing in superposition (published as "Cerebellum as a neural RAM") | robustness from high dimension |

Kanerva's written statements of the same things, with page numbers, are in `KANERVA_TERMS.md` beside this file
(the 1993 chapter, the 1989 Stanford report, the 2009 and 2010 papers, and the 1988 book through a source
that quotes it with its page). The site's terms follow that file.

## The condensed version, in five steps

### 1. A word is a long row of bits, and there are too many of them to keep one location each

The memory stores and reads long patterns of bits: a thousand bits in his lecture examples, ten thousand in his
hyperdimensional computing talks, 32 on the site's small machine. A pattern that long has a space of possible values
no memory can cover one location at a time.

> "you'll never build a memory that has two to the thousand locations because there just isn't enough stuff in the
> universe to build" (WOcYfaXEAFY, 0:47:31)

> "much less in in the brain to build that many locations" (WOcYfaXEAFY, 0:47:42)

### 2. The memory builds a few hard locations, at random addresses, and never moves them

He draws the memory as two matrices beside an address register and a data register, the shape of a computer memory.
The left matrix holds the addresses of the hard locations; the right matrix holds what they store.

> "in this sparse distributed memory model these two matrices are very different have a very different function the
> left matrix is random and it is fixed once you build it it's that's that's it it will not be changed" (WOcYfaXEAFY,
> 0:37:27)

> "each hidden unit is called hard location because you actually have some um physical stuff that represent that
> particular address of memory" (WOcYfaXEAFY, 0:44:30)

> "only some of those addresses in in in that space actually will be represented by a physical memory location by a
> neuron" (WOcYfaXEAFY, 0:47:55)

Why the addresses must never move:

> "when you come if you later on come you know 10 years later come with the same address it would still activate the
> locations that previously got that data and it turns out if you don't have it fixed you're sort of building on sand"
> (WOcYfaXEAFY, 0:52:56)

### 3. A write spreads the word over the hard locations near it

No hard location has exactly the address you give. The address is compared with every hard location's address by
Hamming distance, the count of bits that differ, and the ones within a radius are activated. A write adds the word to
their counters: +1 where the word has a one, -1 where it has a zero.

> "there's almost certainly not going to be a location memory low actually hard location piece of hardware with that
> exact address so what you do is you sort of activate all the memory address addresses and locations that are within
> a certain distance it is this hamming distance" (WOcYfaXEAFY, 0:48:58)

> "writing means that you increment and decrement the contents of those locations" (WOcYfaXEAFY, 0:51:39)

> "whenever there is a minus you subtract whenever there's a plus you add" (WOcYfaXEAFY, 0:41:22)

### 4. A read lets the nearby hard locations vote

A read activates the hard locations near the read-address, adds their counter rows column by column, and keeps the
sign of each sum.

> "reading means that you sum up from the activated locations and then do a threshold function" (WOcYfaXEAFY, 0:51:48)

> "from those activated locations produces a sum column column wise sum and after threshold and you get a binary total
> in the end you get a binary output so that's that's how this simple memory works" (WOcYfaXEAFY, 0:42:50)

### 5. A noisy address still finds the word, and an unknown address gets noise back

A read-address close to a write address activates mostly the same hard locations, so their counters agree and the
stored word comes back. Far away, the activated sets barely overlap and the sum is noise. He calls that the right
answer: a memory should give noise for what it was never given.

> "if two two addresses the writing address so and the reading address if they are very close to each other the
> intersection is large if they're very far from each other the intersection is very small" (WOcYfaXEAFY, 0:52:01)

> "when you have a large intersection you can recreate an old pattern when you have small intersection then it just all
> looks like noise" (WOcYfaXEAFY, 0:52:16)

> "if you're not reading with with with an address that is similar to what you're written into you should be getting
> noise" (WOcYfaXEAFY, 0:52:31)

> "the address can be noisy and still you can retrieve" (NJW2NacQpzM, 0:25:41)

> "you can probe the memory with a noisy address and it can retrieve what actually was stored" (NJW2NacQpzM, 0:25:56)

The best match, and the memory's "I don't know":

> "not only can you find if there's a best match if it is close enough if there's nothing that has been stored anything
> like that you get an answer from the memory i don't know" (GNKbGmXYY0I, 0:55:34)

In the hyperdimensional talks the same memory is the item memory, or cleanup memory, that finds the nearest stored
vector to an approximate one:

> "an item memory or cleanup memory and it finds the nearest neighbor among all known vectors" (1g5VEcnG6fI, 0:58:39)

Iterated reading (using the answer as the next read-address until it stops changing) is in his written sources,
`KANERVA_TERMS.md` row `iterated-reads:` (K2009 p. 144, FKB1989 p. 15); the site's machine does it and the lectures
on disk do not say it in so many words.

## The two comparisons he draws

### A computer's random access memory

> "any single address activates one location when you put the here 20 bits you put 100 000 bits here it activates
> several locations" (WOcYfaXEAFY, 1:09:02)

> "the activated locations are either modified or they contribute to the output just like here the activated location
> is modified when you're storing or the activated location contributes to the output" (WOcYfaXEAFY, 1:09:22)

### The cerebellum

He reads the granule cells, which are enormously many and each receive only a few inputs, as the hard locations, and
the synapses between the parallel fibres and the Purkinje cells as the counters; each Purkinje cell has its own
climbing fibre, which he reads as the write line paired with that cell's read line.

> "resembles the circuit of the cerebellum so Nature has already built a something that actually looks very much like
> the random access memory for high dimensional vectors" (zUCoxhExe0o, 0:28:21)

> "this would give it a very large capacity and independent of the size the vector input vectors and output vectors
> that come from outside the cerebellum" (WOcYfaXEAFY, 1:03:20)

> "the world you learn when you are five years old it still has the same meaning" (WOcYfaXEAFY, 1:13:42)

> "the cerebellum seems to be something like if it is like 100 gigabytes it's like 100 000 books" (WOcYfaXEAFY, 1:17:38)

> "addresses can be noisy and this kind of memory can be made arbitrarily large" (zUCoxhExe0o, 0:27:54)

### Why high dimension makes it robust

> "so the robustness is built in into this kind of system because of the high dimensionality" (hH65_yWSpFc, 0:08:40)

(The words before that line are "the high dimensional reputation is error tolerance": the captioner heard
"reputation" for "representation", so the line is quoted from where the caption is clean.)

## How the site uses this

- THE EXPLAINER (`SETTLE/settle-site/src/sdmexplore/Explainer.jsx`) carries steps 1 to 5 above, each with a picture
  computed from the page's own 32-bit, 64-location machine, and quotes the caption lines marked in its `QUOTES` table.
  `SETTLE/settle-site/tests/sdmexplain.test.mjs` checks that every quoted line is in this file verbatim with its lecture
  id and time.
- THE SDM INTRO (`SETTLE/settle-site/src/sdmexplore/SdmIntro.jsx`) is the one-paragraph version shared by the memory,
  SDMCHAT, SDMPOEM, WEIRD LITTLE SDM GUY, WHAT part two, SDMJEV and results pages; the KANERVA page links to the explainer
  from its HOW AN SDM WORKS section.

## Caveats

- A caption is not a transcript. The captioner mishears names (Purkinje becomes "pukinia"), drops punctuation and
  keeps every hesitation. Where a caption line is quoted it is quoted as it stands, hesitations included.
- The curation channel's titles for WOcYfaXEAFY and hH65_yWSpFc are swapped against their content; the vault index
  records both the published title and what is said.
- No lecture on disk gives the critical distance, the capacity formula or the activation probability in a quotable
  sentence; those come from the written sources in `KANERVA_TERMS.md`.
