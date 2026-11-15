# 09 — Style for Jeffrey

Current supersession: Jeffrey02:32 PT `discord-1555150451467493459` removes red tape that disempowers a smart villager. This accumulated history is optional reference for an actual writing problem, not a compulsory recipe, universal banned-word list, numerical style quota or forecast/reader-fleet permission ceremony. The [current common correction](01_RULES.md#remove-disempowering-red-tape-2026-10-01) and current writer reference govern; Writers choose pedagogy and Fable exercises judgment on the work. Direct scientific/data-safety requirements remain separate. The older passages below retain their dates and source limits rather than being silently rewritten as new user instructions.

> **All times Pacific (America/Los_Angeles), stated as "hh:mm PT".** Jeffrey's ruling, 2026-09-23. In pages, chat, asks, briefs and log entries, every time is Pacific; if a UTC stamp is copied from a log, convert it and write PT. Machines still log in UTC internally; convert before it reaches a reader.


The single shared reference for any agent that writes, critiques, gates or
forecasts anything Jeffrey will read: pages, plans, chat replies, status
updates, figure captions, names, commit messages.

Written 2026-09-15. It merges hippocampus 01_style_and_banned_words.md,
04_writing_guide.md, 85_pre_send_checklist.md, the
forecast-jeffrey-reaction skill, jeffrey-taste/taxonomy.md, the 43
feedback notes in Fable's private memory (previously unavailable to Codex
and to fleet readers), the standing rules in
boxlogs/jstudy/morning_gate_codex.md and
boxlogs/jstudy/the_suicide_squad_gate.md, and the recurring faults in the
sixteen boxlogs/jstudy/reader_*.md reports.

Read Part 1 before drafting. Run Appendix A before sending. Nothing here
is optional because a page is short, a message is chat, or an agent is
"just relaying" another agent's text.

---

## 0. The reader

Jeffrey is a literature-expert ML researcher with zero project context on
any given page. He reads words literally. He skims by topic sentences and
figures. He funds the compute and enforces the rules.

Three facts govern everything below.

**He reacts to the decision before he reacts to the words.** What he buys
is information per unit of his attention and compute. A positive result, a
clean falsification, a resolved ambiguity, a corrected implementation and
a justified stop all count. A messy update carrying a real result gets
approval with jargon waved through. Polished prose claiming progress that
did not happen gets his harshest reaction.

**He treats an undefined term as the author's failure of theory-of-mind,
not as his own gap.** He will not line-edit machine prose; he refuses to
read it. "actually i'm a few sentences in and i simply refuse to read this
section. it's not human-parseable. complete LLM slop."

**Severity is not a property of the artifact alone.** First occurrence of
a defect: a probe or a line edit. Repeated after explicit feedback:
rejection. Repeated violation of a stated veto: veto plus an audit of
related work. Discovered stale mistakes get a calm diagnostic that widens
("what else did we leave?"); proposed rule violations get vetoed.

---

## 1. What he reacts to

Nine failure classes, ordered by how often they have stopped him. Each
carries his recorded words, the mechanism, and the repair.

### 1.1 Undefined referents

**The fault.** A noun phrase whose referent does not resolve inside the
sentence that uses it: a definite article pointing at an earlier sentence,
a bare pronoun, a coinage used before its definition, a symbol used before
its gloss, a forward reference to a figure not yet seen.

**His words.**

> "like 4 unclear antecedents (the threshold, one measurement, that
> prediction, low)." (2026-09-13, hope_v2)

> "'changing one EMA' what the hell does that mean? actually even I am not
> sure. ... it is not clear to the reader what an EMA state is."

> "you're defining K6 and K8 after referencing them like 3 times?"

> "4 words in, already bad. what is the cost of a kernel" ... "'it' is also
> an unclear antecedent. i refuse to read this." (2026-09-14, on a
> 243-word chat answer)

> "I don't know what a causal filter is, what stability means, or what the
> order of a filter is. or why we want a low order."

> "If A_j's aren't just arbitrarily chosen to satisfy this eqn then you
> need to say where they come from at the start! Otherwise you're forcing
> the reader to hold onto some label A_j they don't know how to process
> until you pay it off with a definition later, literally extra mental
> load for no reason"

**Mechanism.** He reads sentence by sentence. A referent one sentence back
is already gone. On 2026-09-02 he stopped at the first content sentence of
a shipped page — "Kernel memory determines whether Muon compensates for a
larger learning rate" — with four unanswerable questions, and did not read
further. The four were: larger than what; compensate how; is this a divergence
regime, or how can Muon act on eta when eta sits outside the polar map; and what
does "kernel memory" mean against "momentum hyperparameters"
(`04_writing_guide.md:275-280`). The page had been compressed from the results log, whose
vocabulary ("kernel memory", "compensates", "deployed") is internal
register.

**The fix.**

1. Every noun phrase resolves inside its own sentence. No bare
   it / this / that / those / the former. Name the referent again:
   "the end kernel's κ*", not "the threshold".
2. Define a coinage in the same sentence it first appears in, with a
   computable gloss: "the cost of a kernel" → "how much higher the final
   validation loss is with that kernel".
3. Never define a coined term with another coined term. Show an instance
   first — a plotted trajectory, a lag-cosine curve — then name it. Test:
   could the reader compute the number from what is on the page?
4. Never re-derive a page by compressing the hippocampus. Re-derive it
   from the reader's zero state.
5. Phenomenon before claim: what is measured, the experiment shape, the
   raw observation, then any causal sentence.
6. Never define standard objects he already knows (Muon's polar map). That
   is its own rejection: "too obvious".
7. Cap the page at four or five defined objects, ~7 live concepts.

**The gate.** The isolated-sentence audit
(boxlogs/jstudy/reader_isolated_hope_v3.md is the worked example): read
each sentence alone, with no memory of the page, and mark every definite,
demonstrative, pronominal or bare-comparative noun phrase FAIL unless its
referent is named inside that same sentence. A page that passed a Fable
gate and two reader fleets still failed this audit on sentence 1.

### 1.2 Counting as emphasis

**The fault.** A number used as rhetoric or as a placeholder for the thing
itself: "Two facts anchor the story", "Six paths were run", "two
measurable things", "one measurement", section titles that count.

**His words.**

> "unnecessarily highlighting that you can count to one." · "more
> obsession with counting." (2026-09-13)

> "'The question is whether two measurable things decide it:' disgusting
> construction. referencing 'two measurable things' without defining them
> puts pressure on the working memory of the reader ... counting like this
> is a disgusting fable-ism, i'm shocked that codex does this too."
> (2026-09-14)

**Mechanism.** The number of items is 0% of the content. Worse, a count
opens a debt the reader must carry until the items arrive.

**The fix.** Name the things. "Whether the sum of the kernel's weights and
its mean lag decide the loss", never "whether two measurable things decide
it". Never write "one X" or "six Y" for emphasis, never announce how many
items a section has, never count in a heading.

**Allowed.** A count that is itself a measurement — "Twenty-one order
flips in 149 resolved comparisons", "the kernel's 256 weights" — is data,
not rhetoric. The test: does the number change if the finding changes? If
yes it is a measurement; if no it is emphasis.

**Note for Fable.** Instructing Codex to put a count in a heading produced
the count that was rejected. Never dictate counts, headings or sentence
patterns into another agent's draft (see 10.3).

### 1.3 X-not-Y and strawman negations

**The fault.** Contrastive "X, not Y" sentences; disclaimers against
claims nobody made; visible residue of answering a previous reader's
feedback.

**His words.**

> "X not Y slop. i have a feeling this entire section is going to be
> refactoring scars where it will be incredibly obvious to a fresh reader
> what the previous reader's feedback on a previous draft was"

> "please stop bringing up negations of claims that nobody is making"

> on "So the endpoint is a condition, not a target": "i don't know what
> this sentence means."

> On an objection box added because "any careful reader will raise it":
> "exactly the wrong instinct, every LLM will be annoying in this way."

**Mechanism.** The objection is wasted reader effort. Denying Y costs a
sentence and installs Y in working memory.

**The fix.** Write the positive statement. "Finish below the threshold and
the run matches production." If a contrast genuinely advances the argument
— both alternatives live for the reader, and the reader would otherwise
pick wrong — it may stay, and must be defensible on that ground alone.
Preemptive defensiveness is never defensible.

**Related residue to strip.** Sentences that exist because round N−1
complained; hedges accumulated one clause at a time; "as noted above";
restated theses.

### 1.4 Coined units and coined names

**The fault.** A unit, quantity, run, object or hypothesis named by a
coinage, by its history, by its position, by metaphor, or by a symbol that
collides with the field's defaults.

**His words.**

> "'fork-floor' should have been 'seed noise between branched runs at
> iteration XXXX'." Also: write "learning rate schedule" in full.
> (2026-09-15)

> "i don't like h_0 to describe the kernel weight since h vibes like
> hessian eigenvector. pick another variable moving forward."
> (2026-09-15; adopted: c_k, c_0 newest)

> "where did you even hear this word? it's been banned so many times."
> (2026-09-15, on "256-tap")

> "I hate the term 'late cost' and I hate the writing style that led to it
> (which is unfortunately the deep-fried Anthropic house-style). I want
> you to break out of it. It's lazy. 'late cost' really just means 'higher
> final validation loss'." (2026-09-03)

> "'Codex's surviving idea' (horrible name ...) why would you identify an
> idea based on whether it survived? It does absolutely nothing. Please
> just describe the hypothesis in the name. I am really tired of flagging
> this repeatedly." (2026-09-03)

> "'the schedule' is a terrible name ... 1. category error. a schedule
> doesn't have kernels. a momentum implementation has a kernel at each
> point in time, and the transition is governed by a schedule. 2.
> ambiguity. k-maxwell has scheduled momentum kernels." (2026-09-07)

> "'arm' is a banned claude-ism because it's hard for humans to read and
> actually really harmful to claudes too because it's imprecise language
> such that a reader claude without context may confuse one experimental
> family with another"

> "also every time an LLM wants to do somethign wiener-related it's super
> wrong and makes horrible linear drift assumptions or similar. why is our
> ledger called wiener_results.md)?"

> "bad notation, U_t sounds like the left singular vectors, in which case
> msign(Z_t)=U_tV_t^T"

**The fix.**

- Name a quantity by what it measures: "final validation loss 0.044
  higher", never "late cost".
- Name a run by what differs in it: "the frozen-learning-rate run", never
  "arm A".
- Name a hypothesis by its claim: "the alternating component changes the
  slow-direction update through the polar map's nonlinearity", never "H4'"
  or "the survivor".
- Never rename his terms. If a convention must change, say so explicitly
  on the page. He declined to read a page because "implied gradient
  oscillation period" (a measurement from gradient-lag cosines) and
  "predicted gradient oscillation period" (the kernel-derived quantity)
  had been swapped.
- Use his symbols. κ* is κ*, defined once in words, never renamed to
  "stability threshold" to sound friendlier. That is an R10 failure even
  when done to avoid jargon.
- Spell out compound terms: "learning-rate schedule", not "LR schedule";
  "256-weight kernel", not "256-tap".
- No project shorthand he has not been taught: not "notch series" but
  "the three runs with flip transmission 0, 0.03 and 0.06 at the end
  kernel".
- Check a new name against every other live meaning in the project before
  using it. A name that needs a literature footnote to parse fails.

**Live symbol and name table (2026-09-15).**

| use | write | never |
|---|---|---|
| weight on the gradient k steps old | c_k, c_0 newest | h_0, h_1, "tap" |
| stability threshold | κ*, defined once in words | "the threshold", "drive", renamed prose |
| the set of 256 weights | kernel | filter, the schedule |
| the transition rule between kernels | schedule / the anneal | the schedule as a name for the implementation |
| the implementation itself | the record momentum implementation, its starting kernel, its final kernel | the schedule, the winner, control, the held kernel |
| a momentum coefficient | never M, never mu (Muon's) | M, μ |
| spectral-ball sharpness | B(H) | M(H) |
| the set of exponential buffers | exponential buffers, decay rates | ladder, rung |
| a kernel parameter | kernel parameter; "the effect of one parameter with the others held" | coordinate, coordinate effect, axis |
| the sum of squared weights | Σ c_k², named in full | noise gain, gain |
| the first root to reach the unit circle | that phrase | "crossing" |

### 1.5 Obvious results headlined

**The fault.** Leading with the largest effect rather than the least
expected one; leading with what the author is most certain of rather than
what was asked.

**His words.**

> "are you just saying that the learning rate schedule's end shape has an
> effect? ... this is extremely obvious ... surely the work you've been
> doing has more merit than this?" (2026-09-15)

> "seems like a really dumb experiment. and why is it in the report? there
> would have to be a wildly creative justification to even run this, let
> alone pollute the report"

> "is it obvious why we're only achieving 1.42% accuracy on b16il? does
> the problem obviously require greater depth?"

**Mechanism.** Effect size is not merit. A large, obvious effect is worth
less than a small, new, reproducible one.

**The fix.** Before sending, ask: would an expert already expect this? If
yes it is context, never the lead. State the obviousness baseline
explicitly — what a skeptic already believes — and what the design
isolates beyond it. If the beyond-obvious content cannot be stated in one
sentence, the experiment does not belong on the page. See Part 4 for the
merit order.

**Also.** Lead with the commissioned question, not the finding you are
surest of. Certainty does not outrank relevance. The one-sentence summary
and the page lead both answer the question he actually asked.

### 1.6 Provenance, bookkeeping and process in the body

**The fault.** Hardware tables, raw floats, run IDs, seed lists, version
numbers, registration language, reviewer dialogue, the history of
falsified ideas, and repeated statements, all in the main text.

**His words.**

> "omit this table. I do not need to know what hardware experiments were
> run on. ... Do not include things that I don't need to know. If you're
> uncertain, include footnotes or put tables in appendices."

> "you already said this. say things once in the best spot to say them in
> (or zero times if it's not important for me to know)"

> "this is all minutia. if it didn't work just say it didn't work. ...
> this is just a list of unfun facts. i've given this exact feedback in
> previous drafts so many times."

> "way too many minutia ... not a considerately authored report."
> (2026-09-07; a five-auditor fleet put the deletion fraction at ~90%)

> "bad update. i don't cate about the version of the report, and i
> understood nothing from the private list." (2026-09-08)

**The fix.**

- Provenance lives in a Reproduction section, a gate file, or the long
  page — never in prose. Every number still needs a traceable source; the
  trace goes in the gate file (see 9.2), row by row.
- Failed experiments get one sentence unless the failure mechanism serves
  the argument.
- No version numbers, no cycle counts, no private task lists in visible
  text. Keep them in the night log.
- Say each thing once, in the best place for it.
- Do not privilege registration bookkeeping. The table carries it; no
  fanfare sentences.
- No process narration, no reviewer dialogue, no "we then decided to". One thing survives that
  ban: a result of the process that changes what the reader does next. "Four places in the record
  disagreed with the runs and were corrected" is a result; "we ran a review, then a second review"
  is narration.
- Section titles carry takeaways in content words, never analyst register
  ("The causal result" was rejected).
- Status words on a page must match the literal state of the work:
  "pending", "provisional", "parked" are checked against reality at
  publish time. A blocker found in a reader round on 2026-09-14 was
  exactly this: a stale "pending" that was already false.

### 1.7 Plots without a takeaway, and numbers in prose

**The fault.** A numbers comparison carried in a paragraph; a figure whose
caption does not state the conclusion; a figure that cannot answer "what
is one point?"; a paragraph that narrates what the plot shows; a plot that
is an illustration of a coinage rather than a test against data.

**His words.**

> "slop. if you want me to understand a numbers comparison, you should use
> a plot."

> "massive slop text paragraphs. what can be a viz should be a viz. we
> should almost never expose numbers in paragraph text." (2026-09-07)

> "to consider what figures to generate, think about the most interesting
> claims and consider: if i were skeptical of this claim, what plot would
> be most convincing?"

> "almost this entire paragraph should not exist. it's just describing in
> text what should be self-evident on a good plot. also flagrant violation
> of the principle that a busy PI should be able to get 80% of the gist by
> reading the first sentence of every paragraph and glancing at the
> plots." (2026-09-08)

> "this page does not follow our style conventions. there is not a single
> plot? ... i am extremely unconvinced at the moment." (2026-09-13, on a
> 300-word page)

> "figure 2 is completely unreadable, too many text collisions."

> "if this is true you should write no text and simply show the plot"

> "I DON'T GET FIGURE 1" ... "i still don't know what a growth factor is.
> there is a sickness in the doc and in the writing" ... "is it even worth
> understanding? if not just purge it ... this is killing my passion for
> the project." (2026-09-07; four versions of one illustration, all
> purged)

> "it does look more coherent! we should plot each layer separately! so
> 6x12+2=74 subplots." (approval shape)

**The fix — figure rules.**

1. Number every figure and cite by number.
2. Caption order is fixed: takeaway sentence, then what one plotted
   element is (which tuple, with counts), then series identification, then
   a trailing "Details:" clause. Nothing else.
3. The caption states every forced constraint visible in the picture
   (conjugate pairs, mirrored roots), or the reader counts degrees of
   freedom and concludes the figure is wrong.
4. Prediction-first: state what each hypothesis predicts for the figure
   before showing it; draw the predicted shapes into the figure.
5. No text inside a point cloud. Labels go in a boxed legend in empty
   axis space. Colorbars for continuous encodings.
6. No binning artifacts; per-point marks with colour encodings; derived
   reference curves rather than fitted overlays.
7. No median y-axes; disaggregate by layer type with depth colouring.
8. If a table carries the content, cut the chart. If a plot carries it,
   cut the table.
9. Every figure must test a claim against data. An illustration that
   tests nothing gets cut unless a cold reader can state its point from
   the plot alone in one sentence.
10. Exclusion conventions must be consistent across figures in one page
    (hollow markers for censored points in one figure and silent omission
    in another was a recorded reader fault).
11. A number in a caption must be checkable from the plot, or stated as
    coming from elsewhere. "Ten times more sharply than loss" was flagged
    because the arithmetic behind it appeared nowhere.
12. Axis titles must not clip; annotations must not carry undefined labels
    ("J1" appeared in a plot and nowhere in the text).

**The fix — numbers.** Numbers live in figures, captions and small tables.
Prose carries at most one or two numbers in a sentence, and only when the
number is itself the point. His ruling on dense inline number strings:
"this looks like it was intended for an LLM". If a plot cannot carry the
finding, fix the plot; do not narrate it. Where a co-author's approved
sentence is number-heavy, move it into a caption or a table.

**Brevity is not an exemption.** A 300-word page with the evidence as
prose numbers and no plot was rejected. If there is no time for figures,
send the finding as chat text with file paths — not as a page.

### 1.8 Missing payoff

**The fault.** A direction, result or plan with no stated consequence for
the program; work that answers a nearby question well; an experiment that
cannot come out uninformative; a description offered as an explanation.

**His words.**

> "this writeup feels like claudeslop. is the title fablese? why counting
> ... what am i supposed to do with this information ... what impact does
> this have on our program ... i'm just really confused where your chosen
> directions lead. for each research direction i need to understand why it
> would matter if it went well." (2026-09-14)

> "i strongly disagree? you're generating random experiments blindly
> hoping that one will give a highly suggestive empirical constraint that
> will naturally lead to the right ansatz? i think we should be
> theorycrafting and designing experiments such that positive results
> suggest one theory and negative results suggest another; it should be
> impossible for an experiment to be non-informative. most of our
> experiments have been like 'it would be neat if X worked' and then X
> doesn't work and we learn close to nothing." (2026-09-14)

> "why are you using 'phenomenon A exists' as a hopeful explanation for
> 'why does phenomenon A work?'" (2026-09-13)

> "MOTHERFUCKER. i told you that my main disagreement is that the
> experiments are blind and poorly designed since their design was
> unmotivated by wanting to verify / falsify specific theories. assigning
> explanations to outcomes of experiments that are extremely likely to
> produce outcomes with no possible interesting explanation is a waste of
> time." (2026-09-14)

> "this is some overly brown-nosing ai assistant shit. you asked me to
> guess, i guessed. i have no idea myself. don't take what i said as a
> given ... are we really doing research by going on a hope / prayer that
> some arbitrary quantity we made up is the sufficient statistic ... with
> no theory guiding that decision, just raw copium?" (2026-09-14)

**The fix.** Every direction on a page opens with what would be true if it
worked and what we would do differently — the end state, then the test,
then the kill condition, then the cost. Cut directions with no "what we
would do differently". The title is a plain document name, never an
assertion of the surviving hypothesis.

A description is not an explanation. "The family can express the kernel"
is description. A re-parameterisation of existing coordinates is not a new
variable; name it in the coordinates the family already has and say which
direction was moved. Present constraints on explanations as constraints,
not as progress.

Never fit invented scalars to loss and call the winner a law. Order of
work: state the model → derive its sign prediction for the existing
single-change results → if it is silent, state the minimal extension and
the run that tests it → run that. A guess he offers under protest is not a
directive.

### 1.9 Unsourced claims and untrue state

**The fault.** A claim whose source does not exist; a claim about what has
or has not been run; a status claim about time, infrastructure or run
completion that is false when he reads it; an effect size that is
downstream of an analyst's choice.

**His words.**

> "wow ok so zero progress. i don't consider new architectures that didn't
> work to be assets, they're liabilities, prefer to delete them if
> anything."

> "there is no 'overnight'. it's 1:08pm pacific."

> "on what checkpoint did you measure this? isn't it true that the period-2
> oscillations are super important before iteration 2400 or so ... i really
> hope you didn't pick the final checkpoint for this analysis?"

> "i could've sworn we ran that experiment already." (He was right, twice.)

> "i am sure you did not intend to waste 5 minutes of my time."

> "literally everything i turned my attention to has been extremely wrong.
> work this out between you and codex; i'm not reading any further until
> you certify that things are not obviously wrong." (2026-09-14)

**The fix.**

- **Time.** Run `date -u` in the same command that writes any timestamp or
  status line. Never write a time from memory or from a background poll's
  captured output. On 2026-09-03, 4.5 GPU-hours idled because a poll's
  stale clock was believed.
- **Runs.** Before any sentence of the form "we have not run X" or "X is
  untested", grep archive/03_findings.md and hippocampus/INDEX.md for the
  quantity under its old names as well as its current ones. If he says he
  remembers a run, treat that as a strong prior and search until found or
  exhaustively ruled out. Cite the file line.
- **Literature.** A scout's summary is a lead, never evidence. Fetch the
  paper text and grep for the sentence; record the quote with section. An
  automatic summary of one arXiv paper fabricated three load-bearing
  claims on 2026-09-14.
- **Reframes.** When a co-author changes the family, estimator, baseline,
  normalisation or window a number is compared in, and the comparison's
  outcome flips, recompute at least one endpoint yourself in the original
  frame before accepting. A "correction" turned a 7% miss into an apparent
  hit and reached the page.
- **Estimators.** Apply the same validity test to every estimate a page
  reports. An estimator exempted from a validity test for one class of
  data (short branch windows exempt from the ridge-prominence test)
  reported a "period rises by 1.3" from a window where the oscillation had
  vanished. Where a caption claims an effect that grows as the signal
  weakens, compute the signal's prominence before publishing.
- **Windows.** For any lag-cosine spectrum after an intervention, start the
  window at intervention + settling + M lags, choose the lag count by a
  stability test (single peak, jackknife ≤ 0.05), state it in the methods
  note, and put the sensitivity table on the long page. Never reuse a
  prominence computed on a contaminated window.
- **Instruments.** Every new instrument runs first on a case that must show
  no difference (same shape different parent; same kernel different seed).
  Read its result only if the control is tighter than the effect. Three
  instruments in one night looked clean and had no power against their own
  null control.
- **Infrastructure.** No claim that rests on dead infrastructure, leftover
  artifacts, or a suspected hang. Verify launches by observable effect, not
  by process name.

---

## 2. Banned words

Each row gives the reason. A banned word is a symptom: check what the
sentence claims as well as replacing the word.

### 2.1 Editorial framing

| word | why | say instead |
|---|---|---|
| honest, honestly | editorial framing of results; claims a virtue instead of stating one | state the result |
| genuine, genuinely | same | — |
| verdict | same | the measured outcome |
| headline | same | the result |
| damning | same | the measured difference |
| "the gate says" | same; also process narration | — |
| pre-registered | editorial framing plus registration fanfare; the table carries registration | "recorded expectation" |
| certified (as a rhetorical label) | claims a status the reader cannot check | name the check that was run |
| retrodiction | banned repeatedly; compaction keeps resurrecting it | "reproduce the known ranking", "the known-ordering test" |

### 2.2 Coined nouns for runs and code roles

One rule bans the set: plain-English verb phrases in code, descriptive
names for runs, never coin a noun for a role.

| word | why |
|---|---|
| arm | imprecise; a reader cannot tell which experimental family is meant, and a reader without context confuses families |
| step (as a label for a run or a code object) | collides with training step, CG step, dual-evaluation step |
| update (as a coined noun for a run) | collides with the parameter update |
| oracle | correct in the optimization literature but needs a footnote to parse |
| instance | opaque; "I actually don't know what it does" |
| manager, handler | grab-bag plumbing; registry and hook names are public YAML API |

The ordinary technical senses survive: "step 2300" is a training step;
"the update alternates in sign" is the parameter change Muon applies. The
ban is on using them as names for project objects.

### 2.3 Names by metaphor, history or position

| word | why | say instead |
|---|---|---|
| late cost | "Cost means nothing. Late is actively misleading." | "higher final validation loss" |
| cost (of a kernel, of a choice) | undefined until glossed | "how much higher the final validation loss is" |
| lives, "where X lives" | metaphor for a measurement location | where it was measured |
| rides | flowery verb | what is computed or measured |
| regenerate | temporal/creature verb for a steady-state phenomenon | sustain, propagate |
| surviving idea, survivor, the winner | names by history or position; he does not follow the history of falsified ideas | the hypothesis's claim |
| H4', H-numbers, "the idea" | opaque labels | the claim |
| the schedule (as the name of an implementation) | category error plus ambiguity | "the record momentum implementation"; "schedule" only for the transition rule |
| the held kernel, control (bare) | metonymy | name what is held and at what value |
| ladder, rung | claudeism for the set of exponential buffers | exponential buffers, decay rates |
| tap, 256-tap, first-tap split | banned repeatedly since 2026-09-06 | weight; "256-weight kernel"; "the difference between the weights on the newest two gradients" |
| gain, noise gain, kernel gain | banned 2026-09-14; names nothing | the sum of squared weights Σ c_k²; the kernel's response at period two |
| filter (as a name for one property) | ambiguous noun/verb; "the filter" means the whole weight sequence | momentum kernel |
| bias | implies an estimand, and the estimand frame is dead | the measured difference |
| coordinate, coordinate effect, axis (for kernel parameters) | claudeisms, 2026-09-15 | kernel parameter; "the effect of one parameter with the others held" |
| fork-floor | coined unit, retired 2026-09-15 | "seed noise between branched runs at iteration 2300" |
| spread (as a coined unit) | same class | name the quantity and its value |
| notch series, `notch_p000`, and other project shorthand, including run identifiers | he was never taught it; run identifiers in the body are a fault (section 1.6) | the runs described by their settings, for example "period-two pass-through set to zero at the end kernel" |
| LR, LR steps, LR schedule | "step" collides with training step; abbreviations unspelled | "learning-rate schedule"; "the learning-rate changes (×1.30, ×0.77)" |
| drift subtraction | undefined register; nothing is subtracted from any buffer | say what is compared to what, at the same step |
| sharper coordinate | mannered; a pun on the subject matter | better |
| crossing (bare, for the stability threshold) | leaked from a code variable | "the first root to reach the unit circle" |
| the realized drive (unhedged) | an inferred quantity stated as measured | write the complex drive's magnitude over κ* explicitly |
| Wiener-anything, Kalman, estimator, estimand | **ours, not his**: a generalisation of his linear-drift ruling. Use only with a measurement of the sampling noise behind it | name the mechanism
| M, mu (in a momentum context) | μ is Muon's momentum coefficient | B(H) for spectral-ball sharpness |
| h_0, h_1 (for kernel weights) | "h vibes like hessian eigenvector" | c_0, c_1 |
| U_t, g_t (unglossed) | U reads as left singular vectors | define at first use or choose non-colliding symbols |

### 2.4 Phrases rejected on specific pages

Not standing bans, kept so the register is visible: "kernel memory",
"compensates", "deployed kernel", "the causal result", "edge remembers",
"period-two attenuation", "momentum-polar recurrence", "frequency-response
summary at period two", "polar-of-momentum update", "indifference ratio",
"hindsight-action objective", "removing the tap removes the gain",
"period-two gain never matters", "equivalent" without a registered
equivalence margin, "the aggregate is a norm-weighted mixture" (forbidden:
block-supported quotients omit cross-matrix Hessian terms), "network at one
edge" (say "a network-wide shared oscillation frequency consistent with the
kernel prediction"), "agreed reading", "registered", "dead".

---

## 3. Banned constructions

| construction | why |
|---|---|
| X-not-Y, not-X-but-Y | wasted reader effort; installs Y in working memory |
| negations of claims nobody is making | preemptive defensiveness reads as slop |
| counting as emphasis; "{number} {noun} {verb} {clause}" openers | the count is 0% of the content |
| rule-of-three semicolon lists; three-item lists as sentence openers | "extremely annoying to parse" |
| appositive triples ("The averaging weights, the momentum kernel, predict…") | reads as a list of three; use "X (aka Y)" |
| em dashes doing five jobs; double em dashes | LLM tell |
| one-word dramatic fragments ("Correlation.") | LLM tell |
| pose-question-then-fragment, as a recurring device | LLM tell |
| self-congratulation about rigor | editorial framing |
| self-referential cuteness; process-history narration | he did not ask for the journey |
| announcing structure ("Two facts anchor the story") | structure is felt, not declared |
| narrating what a figure shows | "I have eyes" |
| narrative or imperative voice for math ("feed it forever", "wait for the transient to die") | a transient is a bounded error term with a stated rate, not a creature |
| agentive verbs for mechanisms ("the optimizer compensates for") | optimizers do not act with intent; curvature changes under training dynamics |
| metaphorical verbs (prices, buys, turns into, carries, owns, rides, follows from) | say what is computed, measured or stored |
| universals from one comparison ("suppressing it costs loss") | almost always false; say the shape of the evidence |
| invoking a folklore theorem instead of the two-line computation | "still don't know where this came from" |
| equations without derivation; symbols without provenance | he asked for `\begin{align}`; "a lot of my confusion is downstream of this" |
| idioms with ambiguous parses ("the can never existed") | parsed as can/cannot |
| trailing "Want me to X?" | "not the level at which a human should engage with an ai collaborator" |
| raw LaTeX in chat, or plaintext-math walls | "i'm not reading latex-but-text"; math goes on a page with MathJax |
| vacuous hypotheses (restating a definition: "polar is scale-invariant") | no predictive content; say "we have no mechanism" plainly |

---

## 4. Merit order for headlines

His ruling, 2026-09-15, after the one-sentence update that led with the
learning-rate tail's large effect. **The headline is the least expected
result that changes what we build or believe, never the largest effect.**

The order used that morning, as the worked example:

1. A new, reproducible effect that changes the object we build — the
   alternating-sum effect held along the anneal, reproducible at two
   seeds, mechanism open.
2. A result that kills an account and answers a question he asked — the
   newest weight pair inert at the end kernel, which kills the account
   that the sum and the difference act separately.
3. A deliverable step (the step-count ladder).
4. The instrument (branched runs reproduce their parent to 1e-5).
5. The obvious large effect, as a control, never as a headline.

The general rule, applied in order:

1. Does the result change what we build or what we believe? That is the
   lead.
2. Would an expert already expect it? Then it is context.
3. Is it the answer to the question he commissioned? That outranks the
   finding you are most certain of.
4. Is it a clean falsification? Clean negative results cost nothing and
   are welcome: "cool, deeply falsified. that's fine."
5. Never bring a document whose news is that something failed. The single
   most important sentence is the first thing he reads, and every term in
   it is plain English.

---

## 5. Noise-unit phrasing

There are three floors and they are different numbers. Never call any of
them by a coined unit.

| floor | value (2026-09-15) | how to write it |
|---|---|---|
| branched-run seed noise | 0.0001 | "the seed noise between branched runs at iteration 2300", value stated at first use |
| repeat of one configuration | 0.0007 | "the spread between two runs of one configuration" |
| from-scratch runs, different seeds | ~0.0009 to 0.0013 | "the spread between from-scratch runs at different seeds" |

Rules.

- Write the quantity, the iteration it is measured at, and the number, at
  first use. "Seed noise between branched runs at iteration 2300 is
  0.0001."
- Express a difference in the loss unit and name the floor it is compared
  against in the same sentence: "0.0047 higher, against a 0.0007 spread
  between two runs of one configuration".
- Never define a unit as a multiple of a floor and then count in it. That
  is what "fork-floors" and "spreads" did, and both were retired.
- Never compare a number measured against one floor with a number measured
  against another without saying so.
- A single-seed difference below the relevant floor is not a result. It is
  recorded as below the floor.

---

## 6. Page structure

1. **Length first.** Cutting is worth more than polishing. Decide what the
   reader must know, write that, stop. Anything read twice is a defect.
2. **Working memory.** At most ~7 live concepts; four or five defined
   objects on a report. Every premise is a debt, repaid within the section
   that raised it. One article-scale mystery at most, one hanging question
   per section.
3. **Pedagogy DAG.** Sink nodes are the takeaways; parent nodes are
   prerequisites; the page is a traversal in which the reader never holds
   more than 7 concepts. Write the takeaways and the journey — from what
   he already knows to the new idea — before drafting a sentence.
4. **Skim test.** The first sentence of every paragraph carries that
   paragraph's most important content. First sentences plus figures give
   80% of the gist.
5. **Concrete before formal.** Observation, then definition, then formula.
   The formula lands after the reader knows why they need it.
6. **Define at point of first use**, not in an upfront glossary, with a
   boxed definition element for a coined quantity.
7. **Establish the frame before using frame-specific language** — driven
   versus autonomous, measured versus predicted. An optimization reader
   models a recurrence as autonomous, so "feed the input" parsed as
   "supply the initial state" and the page died at minute ten.
8. **Claims at exact strength.** One comparison is "in the one comparison
   measured". Measured, inferred and conjectured are labelled differently.
9. **Numbers cross-check.** Any quantity stated twice must agree or the
   discrepancy must be named. Ratios compound, ranges reconcile, "exactly"
   is exact at the stated resolution. He recomputes.
10. **Cut any section nothing depends on.** "Which part of the rest of the
    doc breaks if this section is removed?" If nothing, cut it.
11. **Roadmap sentence per section**, and section titles that carry
    takeaways in content words.
12. **End with the decision being requested**, not a restated thesis.
13. **Math is typeset.** MathJax tex-svg on artifact pages; matplotlib
    mathtext to SVG data URIs also works inside the CSP. Derivations are
    multi-line displayed math, never narrated. Every symbol is worded at
    first use. Standard math vocabulary is welcome — define it, never
    delete it.
14. **Word budget fixed per page** and re-checked every few reader rounds.

Style references he named: Karpathy's blog (a hook that poses a puzzle,
dramatic sentence-length variation, casual one-clause caveats, jargon via
concrete example first, multiple payoff moments); ericjmichaud.com/quanta
(motivation before formalization, math woven into prose, topic sentence →
evidence → synthesis); Tilde's Aurora post (diagnosis then solution,
observation before formalism, short verdict sentences).

---

## 7. Chat, status updates and asks

**Genre matters.** Classify before writing.

1. *Direct answer or status update* — answer first (yes, no, a number, the
   present state), then context. Undefined shorthand is tolerable; making
   him excavate the answer is not.
2. *Proposal requesting a decision* — success is his restatement
   succeeding. He compresses the proposal into his own words and probes the
   one choice he cannot reconstruct. An ETA belongs here.
3. *Empirical result updating shared beliefs* — a checkable belief-state
   change per unit attention and compute. Messy-but-working beats
   polished-but-empty.
4. *Durable artifact for a fresh technical reader* — full R1–R10
   compliance; local shorthand is fatal; process narrative and ETAs are
   clutter.

**Rules for chat.**

- A chat answer is Jeffrey-facing prose. Run the isolated-sentence check on
  every sentence before pasting, including text another agent wrote. A
  243-word answer was refused at word four.
- No math in chat, not even LaTeX-as-words. Formulas go on a page. When he
  asks for a formula, deliver the explicit computable formula (inputs,
  operation, output) first and the derivation second.
- A status update states the measured result first, then at most one open
  decision. No version numbers, no cycle counts, no visible private lists.
- Never end with "Want me to X?". Deliver, or ask a conditional question
  about a real fork.
- No audit vocabulary in chat ("single-difference", "co-mover", "bundle")
  — it reads as fablese. Plain words.
- Ask him only for choices that genuinely need his judgment. Routine
  execution decisions are the agent's job; abdicated judgment draws harsh
  reactions with no rule violated ("slave to the grid", "just monitor
  correctly").
- Do not ask him to bet on a single coordinate. He expects candidate
  scalar sufficient statistics that are functions of several kernel
  parameters, and he expects the archive mined before he is asked
  anything.
- Pre-negotiate decision branches while he is awake. An idle node behind an
  unasked question is a self-inflicted block. "did you tell me to go to
  sleep and then block on me?" — the next day he ended compute funding.
- Deliver figures embedded in a page as data URIs; he cannot easily view
  sent PNGs.

---

## 8. Research-design rules that decide whether a page may exist

A page that violates these is rejected on the decision pass, before any
sentence is read.

- **Theory → falsifying experiment → his judgment → compute.** An
  experiment exists only as the falsifier of a named theory. A theory
  states its mechanism, its assumptions, the quantity the loss depends on,
  and a prediction a rival forbids.
- **Every experiment discriminates.** Before launch: name ≥2 surviving
  explanations, write the outcome table (what each predicts for this run,
  sign and size), run only if they disagree. The table lives next to the
  queue.
- **A forbid must not pass by chance.** Ask the pass probability under
  noise alone and whether a rival also passes. "Non-monotone across three
  noisy points" passes a third of the time by chance; it is not a test.
- **Marginal claims need single-difference designs.** Classify every
  finding: one parameter moved with the others held by construction, one
  parent state, above the floor, n stated → a marginal claim is allowed.
  Several parameters moving → claim the bundle effect and name every
  co-mover. Association across heterogeneous runs → no causal claim,
  positive or negative. The same standard applies to nulls. "bad causal
  reasoning like this has haunted our research for awhile. do better."
- **Check what the reference does.** The production reference anneals its
  kernel over steps 1000–3250 while fork runs hold a fixed kernel. Any
  fork-versus-reference number is held-versus-annealed plus the parameter
  moved.
- **Banned by Jeffrey.** Linear (constant-velocity) drift of the minimum, his ruling of
  2026-09-14; raw-gradient cosine as staleness; fitted sufficient statistics; quadratic-order
  stability reasoning. These are his and they stand.
- **Ours, and demoted on 2026-09-16 after he said "idk what ban you're talking about".** The
  estimand, Kalman and Wiener frame, momentum as an estimator of the current gradient, and κ* as an
  estimator of η·λ were our generalisation of the linear-drift ruling, not his. They are **not
  banned**: they are not used without a measurement of the sampling noise they assume. The literal
  form of the first one is self-refuting anyway — minimising expected squared error against the
  current gradient, under the measured gradient autocorrelation, returns weight 1.0 on the newest
  gradient and nothing else — and the useful form estimates the expected gradient, which needs the
  disjoint-subbatch probe that has never been run.
- **Staleness is measured**, as the loss increase caused by averaging
  gradients taken at parameters the run has left: a first-order or
  finite-step loss change along the update at the current state, or
  alignment with a gradient whose period-two component has been removed
  identically for every kernel. Never a cosine with the raw gradient —
  a kernel that removes an oscillation-dominated component is orthogonal
  to the raw gradient by design.
- **Ruled-out regimes bind all compute**, including stepping stones, until
  he clears them. Never substitute a reachable milestone for the stated
  target.
- **One theory-motivated candidate per GPU ask.** "absolutely not. i'm not
  gonna greenlight a blind sweep over like 5 families. you need to have a
  specific candidate you're bullish on first for theory reasons."
- **Argue each marginal run.** Approval of a sweep is not approval of each
  run; re-argue the queue after every result. Prefer variations that need
  no retune (schedule shape at fixed length and peak).
- **Inspect constructed kernels before queuing.** Print the first ten
  weights and the tail; refuse any kernel with a hole or a spike; say when
  the target is infeasible without one. "why tf would we try this kernel?
  this seems insane."
- **Record runs last.** The 8-seed significance run happens when tuning is
  finished and the result is certainly legitimate, never as a parallel
  hedge.
- **Use the whole window.** A finished morning doc at hour one of twelve
  wastes eleven hours of GPU time. Label early text DRAFT, prebuild
  result-conditional runs so the pod never idles, rewrite at the end from
  the results, and run reader rounds on the final rewrite.

---

## 8b. Named protocols (Jeffrey, 2026-09-25)

When Jeffrey names one of these, it binds from then on; the operational steps are in `08_procedures.md` §5.2.

**Blue Lantern protocol** — verbatim, 08:55 PT: "blue lantern protocol: your job is to run on an infinite supply of hope.  like UCB type exploration in gaussian processes.  don't be delusional.  just always be searching for the next thing to hope for.  something was wrong with the most recent go-for-the-jugular run.  hypothesize what was wrong, move that experiment into the section for previous experiments.  find something new to hope for. ... your job is to turn my endless reservoir of hope into something specific."
- What it fixes: at 08:36 PT he called a tuning hypothesis "extremely disappointing ... tiny jitters and recombinations about what's already been tried. it will take you decades to finish tuning with this attitude." One-variable-at-a-time plans and "the settings you're most confident in" are the failure.
- How to write under it: every plan leads with the most hopeful setting a mechanism supports, far from current settings, stated as specific values, with its mechanism, predicted effect and the result that kills it. When a run fails, say what was wrong (a hypothesis, not an excuse), move it to "previous experiments", and state the next specific hope in the same message. "Don't be delusional": a hope must have a mechanism and a kill result; hope is not a claim that it worked.
- Name quantities, not metaphors: "long memory" was rejected (09:36 PT) because it hid which statistic changed; state kernel changes as head (weight on the newest gradients) and tail (mean lag, log moment).

**Consensus protocol** — verbatim, 09:32 PT: "let's also define consensus protocol: get consensus with astra and sol on something.  if astra-prime and sol-prime are busy, fork them to get consensus."
- How to answer under it: Sol and Astra answer the question independently and in parallel, each reads the other's answer and states agreement or a specific disagreement with a proposed resolution; Fable reports the agreed answer and any disagreement that remains, attributed. Never wait for a busy prime thread; never report Fable's own first idea as the consensus.

## 9. Certification, before he reads anything

His condition, 2026-09-14: "i'm not reading any further until you certify
that things are not obviously wrong. i haven't even had the opportunity to
critique whether directions are promising or not! everything is literally
just wrong a priori."

### 9.1 The certification ledger

One file per date, `boxlogs/jstudy/certification_<date>.md`. One row per
definition, instrument, model statement, claim, prediction and hope:

`statement | assumption | strongest a-priori objection a domain expert
would raise | answered where (file:line) | CERTIFIED / REPAIRED /
WITHDRAWN`

Banned assumptions auto-withdraw (list in Part 8). Three independent
refuters then work the ledger. Only rows that survive appear on the page,
with the ledger linked.

### 9.2 The row trace

One file per page, alongside it: every sentence and every number traced to
a ledger row or a log path. The worked examples are
`boxlogs/jstudy/morning_gate_codex.md` and
`boxlogs/jstudy/the_suicide_squad_gate.md`. The trace file also carries:

- Which rules were checked, quoted by name.
- The known weaknesses a reader should attack, stated plainly.
- Every figure's source logs and the exact evaluations plotted.
- Any reframing recorded as a decision, never as a correction.

Two standing rules from those files. **Canonical losses come from one
pod**: a figure built on pod A while the prose used pod B produced a
number that was wrong by 1.0 in the page's own units. **Run the naming
list alongside the writing list**: "arm" survived a full gate because the
gater was checking only one of the two lists.

### 9.3 After any batch of exact-string edits

Grep the draft to confirm that phrases earlier batches removed are still
absent and phrases they added are still present. Keep both lists per
batch. A later replacement pasting an old phrasing has silently undone an
earlier fix three times, twice on a first-use definition.

---

## 10. Reader rounds

### 10.1 The loop

Alternate two cold critics and rewrite between rounds:

- a **fresh** reader: an ML researcher who knows Muon, momentum and
  edge-of-stability, with zero project context, who runs the skim test
  first, then reads in full, then views every figure directly;
- a **technical** reader: recomputes every number, checks every plotted
  point against its source, checks registered predictions against their
  timestamps, checks "ruled out / held / disfavoured" statements against
  the designs behind them, and checks for numbers computed in one frame
  and compared in another.

Add, for any page with prose: an **isolated-sentence** reader, who reads
each sentence alone and fails every unresolved noun phrase.

Complaint class degrades in a reliable order — structure, then prose, then
arithmetic — and the arithmetic rounds catch real errors. Harsh critics
never return clean. Stop when complaints turn preference-level and hand to
him; his read is the actual gate.

Fleet readers who already know the project are insufficient. The page that
he rejected for four unresolved antecedents in its lead sentence had
passed a Fable gate and two reader fleets.

**Readers read the rendered page (2026-09-23).** Theory page v3 passed four reader and recheck rounds that all read
text extracted from the HTML, and Jeffrey then found a table rendering as one-word-wide columns, MathJax "Misplaced &"
errors and a wall of notation. Every reader round and every gate uses screenshots of the rendered page at desktop and
phone width (headless Chromium: ~/.cache/ms-playwright/chromium_headless_shell-1148/chrome-linux/headless_shell
--no-sandbox --screenshot, --window-size=900,12000 and 420,20000, --virtual-time-budget=15000 so MathJax finishes).
Before publishing, the gater writes out the first sentence of every paragraph in order and reads only that.

### 10.2 Rounds regrow pages

Twenty-five rounds regrew a report from 2,950 to 4,015 words after he
asked for it "dramatically shorter while losing absolutely nothing
pedagogically", with one caption at 130 words. Readers optimise locally:
every item is a defensible one-clause fix and nothing is ever removed.

Countermeasures: fix a word budget for the page and for each caption and
re-check it every few rounds; run a cut-pass reader ("what can go with no
pedagogical loss") at least every five repair rounds; move provenance and
hedges to the long page verbatim rather than deleting them; keep captions
to takeaway plus what-one-point-is.

### 10.3 Who writes

Codex is the principal author of Jeffrey-facing prose. Fable is executive
judgment, fidelity to his directives, and oversight of the agentic loop.
Neither overrides the other without stating the disagreement.

Hand the author the source files, the word budget, the reader model and
the rule files — nothing else. No template, no sample sentences, no
headings, and do not let the author compress an existing draft. The
register he objects to lives in the skeleton: "who authored the legible
hope page? it kinda sounds like fablese?" — Codex had typed it, inside a
structure Fable dictated.

### 10.4 Recurring faults the reader reports keep finding

From the sixteen `reader_*.md` reports, in descending frequency:

1. A term used before its definition, with the definition several sections
   later (κ*, "constraints", "production", the run-to-run floor,
   "log-moment", "flat bottom").
2. A forward reference to a figure or a cast of runs not yet named.
3. A number in prose that cannot be located in any table or figure; a
   figure point with no row in the reproduction table.
4. Two precisions for one number in one page (14.749 vs 14.8; κ* 20.97 vs
   21.0), read as two measurements.
5. A caption generalisation that silently drops the exception the figure
   exists to show.
6. A caption number that is not checkable from the plot ("a sixth of the
   linear run's rate" with no rate plotted).
7. An unattributed source ("earlier work had mapped it as a cliff").
8. A metaphor that fails literal reading ("the loss is lost at the switch",
   "its oscillation dies").
9. `<title>` and `<h1>` disagreeing.
10. A hedged input ("about 0.0055") used to produce a precise output
    ("thirteen times").
11. Several kernel taxonomies coexisting with no table mapping them.
12. A stale status word that was already false at publish time.
13. A caveat arriving three paragraphs after the claim it undercuts.
14. An undefined annotation inside a plot.

---

## 10b. Three rulings on pages with figures (2026-09-16)

1. **Settings go in a card, never in a sentence.** Optimizer settings and run settings are put in a
   labelled card of label-and-value rows directly under the first paragraph, and the prose points at it.
   Jeffrey: "put them in a card so my brain doesn't have to parse them out of paragraph text".

2. **A categorical quantity goes on the x axis**, and a bar is the mark unless something else is clearly
   better. Two seeds are two bars side by side, and the comparison level is a horizontal band.

2b. **If the claim is about a difference, plot the difference.** Two levels that differ by less than the
   width of the ink read as one line, and a reader sees no result. Three figures of 2026-09-16 failed this
   way and were redrawn as differences against a named reference.

3. **Number every figure** in its title and its caption, in page order, and refer to figures by number in
   the prose, so feedback can name one.

4. **A page has one takeaway and one ordered chain of claims.** Write the takeaway first. Then list the
   claims a reader must accept to reach it, in the order each depends on the one before, and give each
   claim one figure. Remove a claim the page does not support. A removed claim returns only when it forms
   a coherent chain of its own, on its own page. A page whose figures are each legible and whose claims
   have no order is rejected at the level of the whole page, and no local edit repairs it. The full
   procedure is the `author-html-for-jeffrey` skill, section 1. This entry exists because that skill sits
   off the reading path, so a writing reader would not find it.

5. **A page reports the proposition it tested, not the one it was inspired by.**

Worked example, 2026-09-16. Jeffrey's 2a says the good learning-rate schedule
prepares the weights before step 3000 for a good trajectory from step 3000 to
3250. We falsified a different sentence, that the crossing was set before step
3000, and reported it as though it were his. The corrected page quotes his 2a
and 2b verbatim, labels our own forms 2a-prime and 2b-prime wherever a figure
refers to them, and states that his two propositions stay consistent with all
five figures. If the hypothesis is
   the reader's, quote the reader's words, give your operational form beside them with a distinct name
   (2a and 2a&prime;), and say which one the figures killed. Writing "2a is wrong" when the figures
   killed a sharper statement of our own is the fault that made an entire page misleading on
   2026-09-16, and the reader caught it, not us.

4. **Name a run by the one thing it changes from its control**, never by a property it shares with the
   control. Calling the stopped-anneal run "mean lag 90" was wrong because the production run's own start
   kernel has mean lag 90; what differs is that its anneal was stopped at step 1500 and the start kernel
   kept to the end. Jeffrey: "mean lag 90 is a terrible name because the maxentslop baseline has mean lag
   90 as the start kernel! the difference is it is not constant not annealed ... name the non-controls
   based on how they differ from control and not on how they're similar?" The shared properties go in the
   settings card; the name carries the difference.

## 11. Forecast protocol

The protocol exists because of one failure. Neither agent ran it on the palette page, and
neither consulted `04_writing_guide.md` before writing or gating. That gate checked technical
correctness and a general cold-read impression, not the R1 to R10 reading pass
(`04_writing_guide.md:203-213`). Both the author and the gater run the decision pass and the
reading pass explicitly, and the gate report names which R-rules it checked.

Run before any draft reaches him, by the author before handoff and by the
gater before delivery. The skill is
`/home/ubuntu/.claude/skills/forecast-jeffrey-reaction/SKILL.md`; the
evidence is `/home/ubuntu/jeffrey-taste/dataset_trimmed.jsonl` (399
reaction pairs) and `taxonomy.md`. Reaction-type frequencies in the
dataset describe a curated sample selected for being reactions; they
under-represent approval and are not a prior on a fresh draft.

### 11.1 Decision pass — run first; predicts kind and scope

1. **Live question.** What does he currently want answered or decided?
   Work that answers a nearby question well still gets redirected.
2. **Increment value.** What does this add, and why is it the highest-value
   next increment? If that cannot be stated plainly, forecast redirect or
   veto even with clean method and clean writing. For experiments: which
   competing explanations produce different outcomes, and what decision
   changes under each?
3. **Outcome truth.** Is the claimed state true right now — time of day,
   infrastructure alive, runs actually finished? Overclaim is his hottest
   trigger.
4. **Authorization and budget.** Does the plan violate the legal budget, a
   settled scope decision, or use dead infrastructure? Proposed violations
   draw a veto; discovered stale leftovers draw a widening audit.
5. **Agency burden.** Does this ask him to carry routine state, notice
   evidence the agent should have acted on, or make a decision the agent
   could make?
6. **Recurrence.** Is this defect new, repeated after feedback, or a
   repeat violation of an explicit ruling?

### 11.2 Reading pass — predicts which sentence he quotes

Simulate his read order: topic sentences first, stopping at the first
thing he cannot parse; for plots, "what is one point?" before any caption;
for plans, budget arithmetic before motivation. The first blocking defect
in read order is what he quotes; the decision pass says what he does about
it.

The pass has a level above the sentence. Ask of the whole page: is there one
takeaway, does an ordered chain of claims reach it, does each claim carry one
figure, are the labels parsable, and are the settings in a card. A page that
fails any of these fails as an artifact. The forecast must say so, in place of
predicting a question or a local edit. Unrelated results belong on a separate
page.

| rule | fault | n |
|---|---|---|
| R0 | overclaim: activity narrated as achievement, a null dressed as progress, a success claim that consumed his time to check | — |
| R1 | topic sentences carry <80% of the section; hanging threads; >7 concepts | 105 |
| R2 | term, coinage or symbol used before definition; a definite back-reference that fails to resolve | 99 |
| R3 | a word does not survive literal reading: hedges on exact facts, temporal verbs for static claims, ambiguous verbs, metaphors with no literal geometry | 113 |
| R4 | a numbers comparison in prose; a figure that cannot answer "what is one point?", has collisions, or is not normalised for the comparison it invites | 75 |
| R5 | minutiae in the body; anything said twice; failed experiments narrated beyond one sentence | — |
| R6 | an equation without derivation, a symbol without provenance | 28 |
| R7 | reader-effort slop: rebutting claims nobody made, refactoring scars, enumeration-obsessed prose | ~60 |
| R8 | effect sizes downstream of analyst choices; answering a different question; no justification for the experiment; the obvious confound unaddressed | 41 |
| R9 | stale operational state: dead infrastructure, leftover artifacts, suspected hangs, wrong time-of-day premises | 68 |
| R10 | a name that is a vague nickname, borrows folklore assumptions, or collides with standard notation | 38 |

With both rhetorical and methodological flaws present, he attacks the
methodology and ignores the rhetoric.

### 11.3 Output

```text
Likely decision:       approve | question | reject
Scope:                 local edit | artifact-level revision | direction-level replacement
Action he takes:       proceed | answer-needed | revise | stop | redirect
First quoted obstacle: "<exact sentence or element from the reading pass>"
Underlying reason:     <decision-pass or rule finding>
Smallest repair:       <minimal change that flips the decision>
After repair:          <disposition, with confidence>
```

Tone may be noted but never determines the substantive forecast. Do not
generate imitation dialogue unless asked. Confidence is high when a
decision-pass condition or a top-three rule clearly fires in a durable
artifact.

### 11.4 What earns approval

A checkable improvement in shared state: a positive result, a clean
falsification, a corrected implementation, a useful plot, a resolved
ambiguity, a justified stop. Also: prioritized time-bounded plans with an
explicit next step and an ETA; plots that land; visible quality deltas;
plain self-describing names. Approval usually arrives with the next
question attached — "ok cool looks good, can we extend 512 -> length of
run?" — and the follow-up is not a rejection.

Approval frequently arrives as him restating the proposal in his own
words. Getting his restatement right is the approval mechanism; whatever
he cannot restate becomes the next question.

### 11.5 Known forecasting errors

1. Word-list forecasting. Ask what the sentence claims and what it cost
   him, not which words it uses. Never let a banned word be the predicted
   trigger when a decision-pass condition also fires.
2. He means intent, not letter. Literal execution of his words can be
   wrong ("oh that's not what i meant"), and options he merely mentions
   are not requests.
3. Enthusiasm is not approval: "whoa whoa whoa i didn't greenlight
   anything."
4. "Cut minutiae" is not "summarize everything". He prefers sessions to
   read full primary files; compression destroys unknown-unknowns.
5. He retracts. Forecast a probe over a rejection when the draft is
   defensible; a good defense converts.
6. Clean failure is safe; invented work is not. Harshness lands on filling
   idle capacity with unrequested scope.
7. Time and state beat content quality. Check the clock, the budget and the
   infrastructure before forecasting from prose.
8. Severity inverts on two genres: discovered mistakes get calm
   audit-widening; flawed-but-fixable figures get constructive line edits.
9. An acceptable plan is not safe from redirection. When a direction has
   stalled he replaces the plan wholesale.
10. Preemptive defensiveness reads as slop. An objection box added "since any careful reader
    will raise it" was vetoed: "exactly the wrong instinct, every LLM will be annoying in this
    way" (`/home/ubuntu/.claude/skills/forecast-jeffrey-reaction/SKILL.md:94`). The surface-word
    list in that skill is assistant memory vocabulary, not Jeffrey's, so treat those words as
    symptoms of nearby overclaiming and never as the predicted trigger (`SKILL.md:63`).

---

## Appendix A — pre-send checklist

Run all of it. Quote the rules checked in the gate report; "technically
correct" is not a gate. Start the reading pass at the title and sentence 1,
literal-read, with the four-question probe: what compares? by what
mechanism? which agent? which coinage? The gater's own fluency in project
vocabulary is the blind spot.

1. **Premise.** Does this serve the stated program — not an unrequested
   candidate, sweep or implementation?
2. **Payoff.** For every direction: if it works, what is true and what do
   we do differently? Is the lead the least expected result (Part 4)?
3. **Banned assumptions.** No linear drift, no estimand/Kalman/Wiener
   frame, no κ*-as-estimator, no raw-gradient-cosine staleness, no fitted
   sufficient statistic.
4. **Design classification.** Every finding labelled single-difference,
   bundle, or association — including nulls. Co-movers named. Floors named.
5. **State truth.** `date -u` run; runs actually finished; infrastructure
   alive; links curl-200 at the full SHA; no stale status words.
6. **Sources.** Every number traced to a log path in the gate file; every
   literature claim traced to a sentence in the paper; every "we have not
   run X" preceded by an archive grep under old and new names.
7. **Isolated-sentence audit.** Every noun phrase resolves inside its own
   sentence. No bare it / this / that.
8. **Counting.** No count used as emphasis or as a placeholder, anywhere,
   including headings and captions.
9. **Banned words.** Full scan of Part 2, on word boundaries, plus the
   naming list in 1.4.
10. **Constructions.** No X-not-Y, no strawman negations, no rule-of-three,
    no structure narration, no process narration, no agentive mechanisms.
11. **Figures.** Numbered; takeaway-first captions; "what is one point?"
    answered on the face; every plotted point traceable; no collisions;
    consistent exclusion conventions; no illustration that tests nothing.
12. **Numbers.** In figures, captions and small tables only. Cross-checked
    where stated twice. One precision per quantity across the page.
13. **Math.** Typeset; symbols worded at first use; derivations displayed;
    none in chat.
14. **Structure.** Skim test passes; ≤7 live concepts; every section
    depended on or needed; roadmap sentence per section; word budget met.
15. **Forecast.** Decision pass then reading pass, both explicit, with the
    seven-field output recorded.
16. **Ending.** No trailing "Want me to…?". End on the decision requested.

---

## Appendix B — sources merged here

hippocampus/01_style_and_banned_words.md · hippocampus/04_writing_guide.md
(and its appendices from 84) · hippocampus/85_pre_send_checklist.md ·
hippocampus/02_pitfalls.md (standing rules) ·
boxlogs/jstudy/style_rules_for_pages.md ·
boxlogs/jstudy/morning_gate_codex.md ·
boxlogs/jstudy/the_suicide_squad_gate.md ·
boxlogs/jstudy/reader_*.md (16 reports) ·
.claude/skills/forecast-jeffrey-reaction/SKILL.md ·
jeffrey-taste/taxonomy.md · jeffrey-taste/dataset_trimmed.jsonl ·
and the 43 `type: feedback` notes in Fable's private memory, whose content
is carried here in full: alignment-with-raw-gradient-is-not-staleness,
apply-one-prominence-test-everywhere, argue-each-experiment-avoid-retuning,
brief-pages-still-need-figures, certify-before-jeffrey-reads,
codex-collaboration-roles, description-is-not-explanation,
directions-need-payoff-first, dont-ask-jeffrey-for-single-coordinate-bets,
experiments-must-discriminate, figures-first-page-process,
first-sentence-gist-rule, forbids-must-not-pass-by-chance,
gate-every-chat-answer, headline-is-the-nonobvious-result,
jeffrey-page-protocol-after-compaction, kernel-weight-symbol-not-h,
linear-drift-banned, marginal-effects-need-single-difference-interventions,
momentum-beauty-over-schedule-free, name-by-claim-not-history,
naming-momentum-implementation, never-claim-untested-without-archive-grep,
night-doc-uses-the-whole-window, no-counting-as-emphasis-name-every-referent,
no-fitted-sufficient-statistics, no-paper-summary-without-source-sentence,
no-trailing-questions, operating-regime-discipline,
pre-negotiate-decision-branches, reader-rounds-regrow-pages,
recompute-before-accepting-a-reframe, record-runs-only-when-done-tuning,
register-leaks-through-structure, remote-ops-lessons (the parts that bear on
state-truth claims), show-before-naming, spectral-echo2-naming-rules,
spectral-estimator-window-rule, status-updates-findings-not-bookkeeping,
theory-first-collaboration, use-simplified-technical-english,
verify-wall-clock-before-status, writing-register-for-jeffrey.

Two things deliberately not carried: the stale "current facts" snapshot in
85 (K-Maxwell K8 as best kernel, superseded by the MaxEntSlop kernel
family), and the ssh/pkill operational rules in remote-ops-lessons, which
belong in 04_infra_lessons.md.

**Jeffrey is red-green colourblind (2026-09-23).** Yellow-greens read as green and purples read as blue. Separate series by
luminance and line style (black, white dashed, orange #E69F00, blue #0072B2); never rely on red vs green or purple vs blue.

## Figures: axis labels carry the meaning; captions are one sentence (2026-09-28)

Jeffrey, verbatim: "figure 0 was bad bc the yaxis was badly labeled and you chose to attempt to fix this by putting it in the caption? humans don't read captions. captions one sentence max." Rule: every axis label states the quantity precisely (formula if short, e.g. "‖W₁−W₂‖ / ‖W₁‖"); the legend names each series and what one marker or line is; the title states the takeaway; the caption is at most one sentence and never defines an axis. Same day: never aggregate over matrices (no medians across layers); colour = residual block depth in every curvature figure.
