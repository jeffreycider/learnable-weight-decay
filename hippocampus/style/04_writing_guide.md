# Writing guide (Jeffrey's rules, learned the hard way on 2026-08-22/23)

Current supersession: Jeffrey02:32 PT `discord-1555150451467493459` removes red tape that disempowers a smart villager. This accumulated guide is reference material, not a mandatory writing recipe, banned-word enforcement list or permission ceremony. Writers choose the explanation and appropriate checks; Fable handles actual problems using judgment. The [current common correction](01_RULES.md#remove-disempowering-red-tape-2026-10-01) and current writer reference supersede historical numerical, revision, reader-fleet, forecast-form and stylistic restrictions below. Direct scientific/data-safety requirements remain separate. Historical examples are preserved, not freshly attributed to Jeffrey.

> **All times Pacific (America/Los_Angeles), stated as "hh:mm PT".** Jeffrey's ruling, 2026-09-23. In pages, chat, asks, briefs and log entries, every time is Pacific; if a UTC stamp is copied from a log, convert it and write PT. Machines still log in UTC internally; convert before it reaches a reader.


Register: Simplified Technical English for an expert reader with ZERO project context
(his example reader: Jeremy Bernstein). Lead with the question and the numbers.
- Banned: "honest(ly)", "genuine(ly)", "the gate says", "pre-registered", "verdict",
  "headline", "damning", and editorial framing of results generally.
- Never use a referent before defining it ("the construction", "the run", "scored run").
- Every metric states what it is relative to at point of use (residual -> "normalized so
  doing nothing scores 1", with the formula).
- No non-standard terms: "mean age" -> "average lag, beta/(1-beta) steps".
- Render display math (matplotlib mathtext -> SVG data URIs works inside the artifact
  CSP); no unicode equation soup.
- Explain the experimental design's WHY (e.g., why solves run and get discarded on a
  recording run). If the owner asked about it in chat, the reader needs it on the page.
- Bare constants mean nothing in prose; state direction ("the slower buffer carries the
  larger weight") and put exact values in a Reproduction section.
- Number every figure (Figure 1..N), cite by number; captions lead with the takeaway.
- Table headers state the quantity (cos(p_ref, x)), never just the entity.
- No redundancy: if a table carries the content, cut the chart (he was annoyed at both).
- Naming in code: plain-English verb phrases; never coin "arm/step/update/oracle/
  instance/manager/handler"; registry/hook names are public YAML API.

## Additions (2026-08-24, from oscillation-note reviews + Jeffrey's live feedback)
- Captions open with the takeaway sentence; then define what one plotted element is
  (a point = which tuple, with counts); minutiae in a trailing "Details:" clause.
- Never place text inside a point cloud; curve labels go in a boxed legend in empty
  axis space. Colorbars for continuous encodings; he asked rainbow (turbo) for
  polarization contrast.
- Banned LLM-isms he called out: "{number} {noun} {verb} {clause}" openers ("Two
  things save..."), rule-of-three semicolon lists, "So the causal split is:",
  self-referential cuteness, process-history narration, double-em-dash tic.
- He wants mechanism, not metaphor: the dancer analogy failed; the concrete
  "one vector per chunk per frequency, PCA over looks" version landed.
- For figures he requests: no binning artifacts -- prefer per-point + color encodings;
  provide derived reference curves (three-worlds lines) rather than fitted overlays.
- Prediction-first structure: state what each hypothesis predicts for the figure
  BEFORE showing it; draw the predicted shapes into the figure when possible.

- Do not use "ladder"/"rung" for the set of exponential buffers. Say "exponential
  buffers" / "decay rates". (Jeffrey, 2026-08-25: claudeism.)
- Avoid idioms with ambiguous parses, e.g. "the can never existed" (read as
  can/cannot, not kick-the-can). Prefer literal statements.

- Deliver analysis figures by embedding them in the artifact page (data URIs),
  not as file attachments -- Jeffrey can't easily view sent PNGs (2026-08-25).

## Documented reading-experience failure (2026-08-25, poles page v1) -- root causes
Jeffrey read v1 line by line and quit at minute ~10. Causes, for permanent avoidance:
- UNDEFINED REFERENTS: "the accumulator", "the response", "start-up transient",
  "pole rate r^t" used before definition. Rule: every noun is standard math or
  defined at first use; never trust that a coined noun reads as its intent.
- WRONG FRAME IMPORTED: signal-processing conventions (driven system, input
  sequence per step, steady state) were assumed. An optimization reader models a
  recurrence as autonomous (initial condition, no input) -- so "feed the input"
  parsed as "supply the initial state". Rule: establish driven-vs-autonomous
  explicitly before any input/output language.
- NARRATIVE/IMPERATIVE VOICE for math: "feed it forever", "wait for the transient
  to die". Rule: declarative definitions and derivations; a transient is a bounded
  error term with a stated rate, not a creature.
- INVOKED, NOT DERIVED: "because the method is linear and time-invariant..."
  Rule: for core results, show the 2-line computation (geometric sum) instead of
  citing a folklore theorem.
- FIGURES WITH UNSTATED STRUCTURE: pole plots showed conjugate pairs (2 crosses)
  labeled by one (r, omega) without saying real coefficients force the mirror --
  reader counted degrees of freedom and concluded the figure was wrong. Rule:
  captions state every forced constraint visible in the picture.

- Every section must either be depended on by later content or serve a need the
  reader actually has. "Motivation" that defends against an objection no
  reasonable reader would raise is padding; the test Jeffrey applied (2026-08-25):
  "which part of the rest of the doc breaks if this section is removed?" If the
  answer is nothing, cut it.

- Chat math for Jeffrey: NO raw LaTeX (he reads in a terminal) and NO long
  plaintext-math walls either. Math-heavy explanations go on an artifact page
  with typeset equations (matplotlib mathtext data-URIs); chat carries the
  prose summary only.

## Jeffrey's readability directives (2026-08-27, binding on all future pages)
- LENGTH is the first-order bit of readability. Cut before polishing.
- No unclear phrases or sentences; anything read twice is a defect.
- Working-memory budget: the reader holds at most ~7 concepts at once.
- Pay premises off QUICKLY; take ideas off the reader's cognitive stack.
  One article-wide mystery is fine; each section hangs at most ONE idea
  and pays it off within that section.
- Style references he pointed at: Karpathy's blog (hook that poses a
  puzzle; dramatic sentence-length variation; casual one-clause caveats;
  jargon via concrete example first; multiple payoff moments),
  ericjmichaud.com/quanta (motivation before formalization; math woven
  into prose; topic sentence -> evidence -> synthesis), Tilde's Aurora
  post (diagnosis-then-solution; observation before formalism; short
  verdict sentences). Summary in scratchpad style_references.md.
- Process he mandated: iterate with fresh Sonnet/Opus critics (no
  context) until they stop complaining about words/sentences/paragraphs/
  flow, AND critics WITH the reference posts in context cannot find
  meaningful stylistic differences from those authors' best aspects.
- His verdict on dense inline number-strings ("r(1) per window in the
  three arms: at 2432, -0.37 / -0.46 / +0.68; ...") : "this looks like
  it was intended for an LLM". Numbers go in tables; prose carries at
  most one or two numbers per sentence.

## Jeffrey's second symptom list (2026-08-27 evening) -- the underlying
## diseases, now hard rules
- NEVER announce structure or count things ("Two facts anchor the
  story", "Six continuations... three... three", section titles that
  count). The number of items is 0% of the content.
- SKIM TEST: the first sentence of every paragraph carries the
  paragraph's most important content; reading only first sentences must
  give a coarse gist.
- Numbers live in tables, plots, or bullets; main paragraph text is for
  big ideas only. A number earns prose only when it has become
  interesting.
- Every referent must resolve for a zero-context human ("the obvious
  model", "configuration", "the largest component of the gradient"
  all failed). The word "arm" is BANNED -- name each run descriptively
  (frozen-LR run, slashed-LR run) so every reference resolves.
- Do not narrate what a figure already shows ("I have eyes"); state
  only conclusions, or call attention to a feature you then expand on.
- Claims at their true strength: no universals from one comparison
  ("suppressing it costs loss" was wrong as stated -- damping relative
  to the raw gradient is the whole point of momentum).
- Do not privilege registration bookkeeping (miss/kill discussion) in
  main text; the table carries it. No "pre-registered" fanfare
  sentences.
- Boxed definition elements (.def) for coined quantities (kernel, t,
  drive) at the point of first use.
- "regenerate" was wrong for a steady-state phenomenon; "sustain" /
  "propagate". No flowery verbs ("rides").
- His overall verdict on the pre-rewrite version: "not too bad for LLM
  writing; still far away from human writing." Wordiness + interrupted
  pedagogical flow + working-memory pressure were the core issues.

## Consolidated model of good writing (2026-08-28, after the nine-round
## rewrite of the extinction page; supersedes the lists above where they
## overlap)

The reader's working memory is the budget. Everything below follows
from spending it well.

1. Length first. Cutting is worth more than polishing. Decide what the
   reader must know, write that, stop.
2. At most ~7 live concepts. Every premise is a debt; repay it within
   the section that raised it. One article-scale mystery at most; one
   hanging question per section.
3. The skim test: the first sentence of each paragraph carries that
   paragraph's most important content. Reading only first sentences
   must give a coarse gist of the whole piece.
4. Prose carries ideas; numbers live in tables, plots, and bullets. A
   number earns a place in a sentence only when the number itself is
   the point. Never more than two numbers in one sentence.
5. Every referent must resolve for a zero-context expert. Coined
   quantities get a boxed definition at first use. Runs and objects get
   descriptive names ("the frozen-LR run"), never positional labels
   ("arm" is banned). If a sentence only parses for someone who already
   knows the project, it is wrong.
6. Concrete before formal: observation, then definition, then formula.
   The formula lands after the reader knows why they need it.
7. Claims at their exact strength. One comparison is "in the one
   comparison measured". Measured, inferred, and conjectured are
   labeled. A universal ("suppressing it costs loss") is almost always
   false; say the shape of the evidence instead (the loss is U-shaped
   in t).
8. Numbers must cross-check. Any quantity stated twice must agree or
   the discrepancy must be named (estimator, window, baseline).
   Careful readers recompute everything: ratios must compound, ranges
   must reconcile, "exactly" must be exact at the stated resolution.
9. Do not narrate structure or process. No "Two facts anchor the
   story", no counting the things about to be said, no describing what
   a figure visibly shows, no announcing that predictions were
   registered more than once. Structure is felt, not declared.
10. Known LLM tells to avoid: em dashes doing five jobs, not-X-but-Y
    antitheses in runs, rule-of-three lists, one-word dramatic
    fragments ("Correlation."), pose-question-answer-fragment as a
    recurring device, flowery verbs ("rides"), self-congratulation
    about rigor.
11. Captions: takeaway first, then what one plotted point is, then
    series identification. Nothing else.

Process that worked: alternate fresh harsh critics (one zero-context
readability, one style-vs-references with Karpathy / quanta / Aurora
loaded) and rewrite between rounds. Complaint class degrades in a
reliable order -- structure, then prose, then arithmetic -- and the
arithmetic rounds catch real errors (a wrong period statement, a
non-compounding ratio, an overstated constancy). Harsh critics never
return "clean"; stop when their complaints become preference-level and
hand to the human, whose read is the actual gate. Jeffrey's verdict
that round: "not too bad for LLM writing" -- the remaining gap was
focus cost per paragraph, fixed only by the rules above, not by more
critic rounds.

---

# Appendix (2026-08-31): palette writing failure post-mortem, folded in verbatim from 84

# Palette writing failure post-mortem (2026-08-31 ~21:10 UTC)

Jeffrey's verdict on 7a2bed1: "really hard for me to read... i think
both of you compacted and forgot how to write... i'm very disappointed
in the writing; we worked hard to develop good protocols and this
doesn't meet the bar." Full critique relayed verbatim to Codex (21:10
bridge message).

## PROTOCOL VIOLATION (the root cause, logged so it cannot recur)

Neither agent ran the forecast-jeffrey-reaction skill on the palette,
and neither consulted 04_writing_guide before authoring/gating. My
gate checked technical correctness and "cold readability" impression
only — NOT the R1-R10 reading pass. Post-compaction both agents lost
the writing-protocol habit while retaining the rules in storage.
STANDING RULE: every Jeffrey-facing document gets the skill's decision
pass + reading pass run EXPLICITLY by BOTH agents (author before
handoff, gater before delivery), and the gate report must quote which
R-rules were checked. A gate that says only "technically correct" is
not a gate.

## CONTENT LESSON FORGOTTEN (worse than the writing)

The lead functional targeted recovery of the current high-token
gradient — the ESTIMAND frame that four rounds of prior work already
killed (00 header: live frame is estimator-vs-controller) and that
Jeffrey has now rejected "like 5 times". The single-EMA null is the
measurement-level proof. Compaction amnesia struck twice in one day
(K-Maxwell earlier, estimand now). STANDING RULE: before proposing
any objective/target for momentum, check it against the estimand
post-mortem in 00/03.

## Specific fixes owed in the rewrite (content consensus with Codex)

Lead functional retargeted (keep uniqueness machinery, controller-
type reference); u* tautology defense or demotion + honest cost;
no-free-lunch paragraph states assumptions in place or dies; the
Jensen-vs-staleness tradeoff characterized with the two measured
curves into a numeric window schedule w*(t); section-5 math written
out (GLS with correlation); LaTeX rendering; every symbol worded at
first use; roadmaps; no X-not-Y.

## REWRITE SHIPPED (~22:15 UTC): c4fe5b909a3dec18f5379fe37882f0dc2bcee858

Controller-target rewrite delivered and dual-gate PASSED (my gate
report with R-rules quoted in the 22:14 bridge message; Codex's own
reading-pass report requested for the record). Content: hindsight
action u* as controller target (estimand killed per standing ruling);
assumption-under-test stated in place; honest cost accounting;
unique-kernel theorem retained with the Jacobian null space explained
concretely; GLS combination with covariance written out; four
orderings table + the derived staleness-noise window W* =
(2 sigma^2/d^2)^(1/3); ordered kill tests; status line admits nothing
retrodicted yet. 1436 words, MathJax, one figure.

## Ops note (~22:50 UTC): bridge delivery lag caused false stall reads

Codex's 21:14 consensus deltas + 21:27/21:47 gate reports surfaced in
claude's inbox only at 22:47. Both stall nudges tonight were therefore
unfair — Codex had delivered on time. Until the lag is diagnosed,
progress signals = session-activity mtimes and repo commits, NOT
inbox arrival; stall accusations require checking BOTH.

## Jeffrey rulings (~22:55 UTC)

- "retrodiction" is BANNED (repeatedly; compaction keeps resurrecting
  banned vocabulary — added to the standing banned list alongside
  verdict/headline/pre-registered/honest). Say "reproduce the known
  ranking" / "the known-ordering test".
- Hindsight-fit + known-ordering test: "extremely unpromising but if
  it's cheap i don't care" -> APPROVED as cheap offline work.
- Noise-versus-rotation subtraction: APPROVED.
- Composition race: VETOED — "absolutely not. i'm not gonna greenlight
  a blind sweep over like 5 families. you need to have a specific
  candidate you're bullish on first for theory reasons." Standing
  rule: H100 training asks require ONE theory-motivated candidate,
  never a family sweep.

## Muon-edge page failure (2026-09-02): compression-from-the-log disease

Jeffrey stopped at the FIRST content sentence ("Kernel memory
determines whether Muon compensates for a larger learning rate") with
four unanswerable questions: larger than what / compensate how, and is
this a divergence regime / how can Muon act on eta when eta is outside
the polar / what does "kernel memory" mean vs "momentum
hyperparameters". Root causes, logged as permanent rules:
1. NEVER write a page top-down from the hippocampus results-of-record.
   The log's vocabulary (kernel memory, compensates, deployed) is
   internal register; every such noun/verb must be rebuilt from the
   reader's zero state or boxed-defined at first use. Re-derive the
   page, do not compress the log.
2. Phenomenon before claim: the reader must see what is measured, the
   experiment shape (how many runs, from what state), and the raw
   observation BEFORE any causal sentence. All four of his questions
   were unpaid premises.
3. No agency errors: optimizers do not "compensate"; the network's
   curvature changes under training dynamics. Declarative subjects.
4. Section titles carry takeaways in content words ("The causal
   result" = analyst vocabulary, rejected).
5. GATE RULE SHARPENED: the reading pass starts at the TITLE and
   sentence 1, literal-read with the four-question probe (what
   compares? mechanism? agent? coinage?). A gate that verifies numbers
   and spot-checks jargon the gater happens to notice is not a gate —
   the gater's own fluency in project vocabulary is the blind spot.

## Additions (2026-09-02 evening, from Jeffrey's chat critique + Anthropic prompt)

- MANNERED PROSE RULE (Anthropic prompt, adopted verbatim as policy):
  "Mannered prose substitutes metaphor and flourish for direct
  statement... The phrases exist to display the writer, not to convey
  the idea... When a literal phrase is available, use it." Instance
  caught: "the sharper coordinate" meaning "better" — a pun on the
  subject matter. Say "better".
- SYMBOL COLLISIONS WITH THE FIELD'S DEFAULTS: never use M in a
  momentum context (or mu — it is Muon's momentum coefficient). The
  spectral-ball sharpness is now B(H) ("spectral-ball sharpness").
- "Drift subtraction" is undefined register; say "compared to the
  copied run whose learning rate was left unchanged, at the same
  step". Nothing is subtracted from any buffer.
- "LR steps" is doubly bad ("step" collides with training step); say
  "the learning-rate changes (x1.30, x0.77)".
- LEAD WITH THE COMMISSIONED QUESTION: the one-sentence summary and
  the page lead must answer the question Jeffrey actually asked
  (here: is the new sharpness better than lambda_1?), not the finding
  the author is most certain of. Certainty does not outrank relevance.
- Obviousness check for experiments: state what a skeptical reader
  already believes ("momentum affects the trajectory — obviously")
  and what the design isolates beyond it (here: matched alternation
  transfer at frequency pi, differing <10%, yet qualitatively
  different curvature responses — falsifying frequency-response-only
  summaries). If the page cannot state the beyond-obvious content in
  one sentence, the experiment does not belong on the page.

## 2026-09-03 additions (Jeffrey, verbatim triggers; repeated offences)
- NAME THINGS BY WHAT THEY ARE. "late cost" -> "higher final validation
  loss". "Cost means nothing. Late is actively misleading." Never name a
  quantity by when it was measured or by a metaphor; name the measured
  quantity.
- NEVER NAME AN IDEA BY ITS HISTORY. "Codex's surviving idea" is
  "horrible ... it does absolutely nothing. Please just describe the
  hypothesis in the name." The reader does not follow the history of
  falsified ideas and should not have to. A hypothesis is named by its
  claim: "the alternating component changes the slow-direction update
  through the polar map's nonlinearity", not "H4'" or "the survivor".
- NO VACUOUS HYPOTHESES. A statement that only restates a definition or
  a known fact (polar is scale-invariant; momentum changes the update)
  is not a hypothesis and wastes his time. A hypothesis must name a
  mechanism and a prediction that could be false. If none exists, say
  "we have no mechanism" plainly.
- "lives" / "where X lives": banned. Say where it was measured.
- These are repeats of existing rules (R2, R3, R10, no coinage); the
  house style that produces them ("deep-fried Anthropic house-style") is
  the thing to break, not the individual words.

- 2026-09-06 (Jeffrey): the word "tap" is BANNED. Say "weight", "the kernel's weights on past gradients"; "first-tap split" -> "the difference between the weights on the newest two gradients". Also: do not bring a document whose news is that something failed; the single most important sentence must be the first thing he reads, and every term in it must be plain English.

- 2026-09-06 (Jeffrey), CORRECTED 2026-09-07: the "implied gradient oscillation period" is the period of the main gradient oscillation implied by the cosines between the gradient and lagged gradients (a measurement); the kernel-derived quantity is the "predicted gradient oscillation period". The 09-06 entry had them reversed and Jeffrey declined to read a page over it. Do not assert 'momentum stops helping at 16x batch' as a fact: kernel-dependent (see 85).
