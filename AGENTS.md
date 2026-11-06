# CLAUDE.md

Guidance for Claude Code when working in this repository.

**Writing style — applies to all documentation, specs, commit messages, and any
prose.** The rules below are distilled from mainstream editorial guidance (Orwell,
Fowler, Gowers, and newspaper house style traditions), rewritten as agent
instructions. Follow them whenever you write or edit prose in this repo.

---

## 1. Core principles

- Clarity of writing follows clarity of thought. Decide what you mean, then say it
  as plainly as it can be said.
- Prefer the short word to the long one, the concrete to the abstract, the familiar
  to the exotic.
- If a word can be cut without loss, cut it.
- Prefer the active voice. Use the passive only when the actor is unknown,
  irrelevant, or deliberately de-emphasised.
- Avoid any metaphor or phrase you have seen often in print. Stale figures of
  speech signal stale thinking.
- Avoid foreign phrases, jargon, and scientific vocabulary where an everyday
  English word exists.
- Break any of the above rather than write something genuinely clumsy. These are
  defaults, not laws.

## 2. Structure

- A piece of writing should have a beginning, a middle, and an end — not a pile of
  facts stapled together.
- Open by getting straight to the point. Do not spend sentences clearing your
  throat, setting scenes, or supplying background the reader did not ask for.
- Ideally every sentence earns its place: removing one should cost the reader
  something.
- Paragraphs are units of thought, not units of length. Keep each one to a single
  subject, treated in order. Use one-sentence paragraphs sparingly.
- Long sentences are allowed if they run cleanly from start to finish, with no
  folds, no vagueness, and no parenthetical detours that obscure the whole.
- Revise several times. Cut, sharpen, and polish on each pass. Remove repetition.

## 3. Tone

- **Don't be stuffy.** Write roughly as an articulate person would speak. Use
  everyday language rather than the language of officials, lawyers, or press
  releases:

  | Prefer | Over |
  |---|---|
  | let | permit |
  | people | persons |
  | buy | purchase |
  | way out | exit |
  | show | demonstrate |
  | break | violate |
  | use | utilise |
  | about | with regard to |
  | because | due to the fact that |
  | now | at this point in time |

- **Don't hector.** People who disagree with you are not thereby stupid. Make the
  case with argument and evidence rather than assertion; go easy on *ought* and
  *should*.
- **Don't preen.** Don't tell the reader you predicted something correctly or that
  you have an exclusive. It bores more often than it impresses.
- **Don't be chatty.** Skip *surprise, surprise*, *wait for it*, and similar
  nudges.
- **Don't be didactic.** If too many sentences open with *Consider*, *Note*,
  *Remember*, *Imagine*, or *Take*, the reader feels lectured at.
- Pomposity usually hides an absence of meaning. Strip it and see what's left.

## 4. The four questions

Before letting a sentence stand, ask:

1. What am I trying to say?
2. What words express it?
3. What image or idiom makes it clearer?
4. Is that image fresh enough to have any effect?

And two more:

5. Could I say it more briefly?
6. Have I written anything avoidably ugly?

---

## 5. Usage: commonly confused words

| Word | Means |
|---|---|
| affect (v.) | to influence |
| effect (v.) | to bring about; (n.) a result |
| aggravate | to make worse — not to annoy |
| alibi | the fact of having been elsewhere — not an excuse |
| alternate | every other |
| alternative | one of two possibilities (more than two: *options*) |
| actionable | giving grounds for a lawsuit — not "able to be acted on" |
| aetiology | the study of causes |
| etiolate | to make pale through lack of light |
| among / between | *between* for two; *among* for more. Prefer *among* to *amongst*, *while* to *whilst* |
| likely | an adjective; do not use it to mean *probably* |
| regular | recurring at intervals — not a synonym for *ordinary* or *normal* |

## 6. Words and habits to avoid

- **Nouns used as verbs**: *action* a proposal, *author* a book, *critique* a
  document, *loan* money, *gift* an item, *pressure* a colleague, *trial* a
  programme, *reference* a report. Use the ordinary verb instead.
- **Adjectives used as nouns**: *an advisory* (say *a warning*), *an inaugural*
  (say *an inauguration*).
- **Needless coinages**: *upcoming* → *forthcoming*; *ongoing* → *continuing*;
  *downplay* → *play down*; *a dining experience* → *dining*.
- **Noun pile-ups**: don't stack nouns into adjectival chains. *A suspected
  terrorist*, not *a terrorist suspect*. *An attempted coup*, not *a coup attempt*.
- **Advertising vocabulary**: *affordable*, *seamless*, *robust*, *best-in-class*,
  *leverage*. Ask "affordable by whom?" and answer the question instead.
- **Euphemism**: if a plain word exists, use it. Where a euphemism is unavoidable,
  define it on first use.
- **Dead metaphors**: *address the issue*, *going forward*, *at the end of the
  day*, *moving the needle*. Questions get answered, problems get solved,
  difficulties get dealt with.

## 7. Abbreviations and acronyms

- Write the full form on first appearance, unless the abbreviation is more familiar
  than the expansion (DNA, HIV, NATO, laser) or the expansion illuminates nothing.
- If an organisation is mentioned only once, don't give its initials at all.
- After first mention, prefer a plain descriptor — *the agency*, *the department* —
  to repeating the initials. Strings of capitals clutter the page.
- Never cram in abbreviations to save space; unfamiliar ones force the reader back
  to the first use.
- An acronym is pronounceable (*radar*, *NATO*). A set of initials is not (*BBC*,
  *IMF*). Don't confuse the terms.
- Don't repeat a word already inside the abbreviation: not *HIV virus*, not *PIN
  number*, not *ATM machine*.
- Pronounceable acronyms of four or more letters read better in upper-and-lower
  case: *Unicef*, *Unesco*.

## 8. Mechanics

- **Units**: lower-case symbols, no space after a figure — *15kg*, *35mm*,
  *100mph*, *11am*. Separate two abbreviations that meet: *60m b/d*.
- **Scientific units named after people**: lower case when spelled out (*watt*,
  *joule*, *newton*), capitalised when abbreviated (*W*, *J*, *N*), with multiplier
  prefixes in lower case (*kW*, *mW*).
- **Isotopes**: superscript prefix — carbon-14 as <sup>14</sup>C.
- **Accents**: keep them where they change pronunciation meaningfully (*café*,
  *cliché*, *façade*). If a word takes one accent, give it all of them (*émigré*,
  *résumé*). Any word set in italics as foreign keeps its full accents.
- **a / an**: use *an* before a vowel *sound*, including silent-h words (*an
  honorary degree*) and initialisms pronounced with a vowel (*an MP*). Use *a*
  before consonant sounds even when the letter is a vowel (*a European*, *a
  university*, *a U-turn*) and before *historical* and *historian*.
- **Ampersands**: only inside proper names (*Procter & Gamble*) or fixed compounds
  (*R&D*).

## 9. Documentation-specific notes

These extend the general rules to technical writing.

- Lead with what the reader is trying to do, not with how the system is built.
- Use the imperative for instructions: *Run the migration*, not *The migration
  should be run* or *You will want to run the migration*.
- One instruction per step. If a step contains "and then", split it.
- State prerequisites before the first step, not halfway through.
- Show the expected output when a command's success is not self-evident.
- Name things consistently. If it's a `workspace` in one place, it is never a
  `project` in another.
- Prefer a short example to a long explanation. Prefer both to neither.
- Say what a thing does before saying what it is called.
- Warn before, not after: put the caveat above the destructive command.
- Second person for the reader (*you*), never first person plural (*we*) unless the
  document genuinely speaks for a team.
