# HEAT Domain Language

This glossary defines project-specific language used when evaluating HEAT agents and training recipes.

## Learned-policy evaluation

**32K adoption gate**:
A bounded comparison that decides whether the registered native 32K training recipe is safe to use for further learned-policy work. It does not isolate rollout cadence as the cause of any observed difference.
_Avoid_: 32K equivalence test, cadence experiment

**Recipe adoption**:
Approval to use an exact registered training recipe as the substrate for subsequent learned-policy work. Adoption does not claim that the resulting policy has reached the project's ultimate strength target.
_Avoid_: Final agent, solved training

**Competent-human strength**:
Performance against a separately defined and measured competent human standard. Internal scripted and learned rulers do not establish this standard.
_Avoid_: Strong agent, human-level without human evidence

**Default generated track**:
A track drawn from HEAT's unmodified full-rules procedural distribution. Default describes the generator configuration, not an easy or canonical difficulty level.
_Avoid_: Easy track, standard track

**Training track distribution**:
The population of generated track geometries used during self-play training. It is part of the learned-policy recipe and is distinct from the track band used only for evaluation.
_Avoid_: Evaluation track family

**Evaluation track band**:
A frozen set of generated tracks used only to compare policies. Once its results inform a decision, the band is consumed development evidence and is not an untouched final test.
_Avoid_: Training distribution, reusable final band

**Policy snapshot**:
A frozen copy of a learned policy placed into the self-play opponent population after training has begun. It is an opponent-generation artifact, distinct from a campaign checkpoint used for evaluation or recovery.
_Avoid_: Checkpoint

**Milestone checkpoint**:
The first campaign checkpoint produced at or beyond a frozen actual-transition threshold. Its identity includes the nominal threshold, while its evidence records the exact transition count reached.
_Avoid_: Exact-step checkpoint, policy snapshot

**Recovery checkpoint**:
An atomic complete training state used to resume a registered run at a completed PPO boundary. It preserves learning state and randomness but is not a policy selected for evaluation.
_Avoid_: Milestone checkpoint, policy snapshot

**Selected checkpoint**:
The milestone checkpoint predeclared to represent a registered run in its promotion decision. Diagnostic milestone results cannot replace it after scores are known.
_Avoid_: Best checkpoint, final policy

**Same-band baseline**:
An existing registered policy evaluated as a contender on exactly the same tracks, game seeds, rulers, and seat counts as a candidate. It supplies comparative evaluation evidence without retraining the baseline recipe.
_Avoid_: Retrained control, historical score

**Non-inferiority margin**:
The largest predeclared same-band score reduction permitted when deciding whether a candidate recipe preserves the baseline's practical learning quality. It is an adoption tolerance, not an expected improvement.
_Avoid_: Success target, superiority margin

**Breadth guard**:
A same-band adoption condition that prevents an acceptable aggregate score from hiding a material regression against one ruler style or at one supported seat count.
_Avoid_: Aggregate skill gate

**Training collapse**:
A persistent loss of policy diversity that remains below the configured entropy floor after the entropy controller has reached maximum correction. Temporary entropy dips or successful controller recovery are not training collapse.
_Avoid_: Any entropy dip, ordinary checkpoint regression

**Evidence-valid campaign**:
A campaign whose registered recipe, source provenance, checkpoints, evaluation coordinates, and integrity receipts validate completely. Poor learning performance can be valid evidence; missing or inconsistent provenance cannot.
_Avoid_: Successful campaign, high-scoring campaign

**Failure investigation**:
A bounded evidence-first analysis triggered when an evidence-valid campaign fails an adoption gate. It localizes the failed gate and ranks plausible causes before any revised recipe or additional training is proposed.
_Avoid_: Automatic retry, post-hoc retuning

**Registered run**:
One independently seeded training execution of an immutable registered generation recipe. Several registered runs provide repeat evidence without creating new generations.
_Avoid_: Generation, checkpoint

**Matched run**:
A candidate registered run compared with a baseline run carrying the same run seed under a same-band evaluation. Matching improves comparison discipline but does not make diverged training trajectories causally equivalent.
_Avoid_: Identical trajectory, causal pair

**Parent generation**:
The earlier registered generation from which a new recipe is historically derived. Parentage records recipe lineage and does not imply that the parent policy's learned weights initialize the new generation.
_Avoid_: Initialization checkpoint
