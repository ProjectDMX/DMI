-------------------------- MODULE VersionAllocator --------------------------
(***************************************************************************)
(* A TLA+ model of the DMI "sole-claimant" catalog version allocator.      *)
(*                                                                         *)
(* SOURCE OF TRUTH                                                         *)
(*   native/csrc/catalog/version_allocator.cpp:49-89  (allocate_version)   *)
(*   native/csrc/catalog/version_allocator.h:5-7      (the claim we check) *)
(*   native/csrc/catalog/clickhouse_client.cpp:107    (deciding_read)      *)
(*   native/csrc/catalog/catalog_writer.cpp:587-620   (watermark publish)  *)
(*                                                                         *)
(* The C++ loop, verbatim in structure:                                    *)
(*                                                                         *)
(*   for (attempt = 0; attempt < allocation_attempts; ++attempt) {         *)
(*     claimed   = max_version("capture_version_claims","version");   //:53*)
(*     floor     = max(claimed, max_version("index_watermark",...));  //:54*)
(*     spread    = attempt == 0 ? 0 : rng() % (8*attempt + 1);        //:59*)
(*     candidate = floor + 1 + spread;                                //:63*)
(*     INSERT (candidate, claim_id) with insert_quorum;               //:66*)
(*     owners    = SELECT claim_id WHERE version = candidate;         //:73*)
(*     if (owners == {claim_id}) return candidate;                    //:81*)
(*   }                                                                     *)
(*   throw CatalogError(kAllocation, ...);                            //:86*)
(*                                                                         *)
(* Both reads carry deciding_read() = select_sequential_consistency=1.     *)
(* The whole point of this spec is to ask what that setting buys and what  *)
(* it does NOT buy.                                                        *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Allocators,   \* the concurrent allocator processes (one per driver process)
    MaxVersion,   \* version ceiling -- a STATE-SPACE BOUND, not in the code
    Attempts,     \* AllocatorConfig::allocation_attempts (16 in production)
    MaxSpread,    \* bound on the jitter; the code's bound is 8*attempt
    Mode          \* store consistency model, one of:
                  \*   "Linearizable"        -- a read sees every insert that
                  \*                            completed before it. What
                  \*                            select_sequential_consistency=1
                  \*                            is claimed to buy.
                  \*   "EventuallyConsistent"-- each reader has its OWN stale
                  \*                            view: two allocators can be
                  \*                            reading different replicas at
                  \*                            different points.
                  \*   "SharedStaleFrontier" -- a weaker-than-linearizable but
                  \*                            SINGLE global lag: all readers
                  \*                            share one prefix. Kept because
                  \*                            it is a strictly stronger store
                  \*                            than #2 and the results differ.

(***************************************************************************)
(* One claim_id per (process, attempt): the code mints a fresh uuid_v4 on  *)
(* every attempt (version_allocator.cpp:64), so ids never repeat.          *)
(***************************************************************************)
ClaimIds == Allocators \X (0 .. Attempts - 1)

VARIABLES
    at,        \* ClaimIds -> 0..MaxVersion. The append-only claims table.
               \* at[c] = 0 means "this row was never inserted".
    seen,      \* Allocators -> SUBSET ClaimIds. PER-READER visibility: the
               \* rows THIS allocator's deciding reads can observe.
               \* Linearizable: an insert enters every allocator's view in the
               \* same step, so seen[a] = Inserted for all a, always.
               \* EventuallyConsistent: a row is accepted (lands in `at`) but
               \* enters one allocator's view at a time, via Reveal. Each
               \* reader therefore observes its own arbitrary subset of the
               \* accepted inserts.
               \* SharedStaleFrontier: Reveal adds the row to EVERY view at
               \* once -- one global lag instead of per-replica lag.
    pc,        \* Allocators -> control point inside allocate_version
    attempt,   \* Allocators -> the loop counter, 0-based like the C++
    cand,      \* Allocators -> the candidate currently being tried
    ret,       \* Allocators -> value returned by allocate_version (0 = none)
    retPubMax, \* Allocators -> max(index_version) published at the INSTANT
               \* this allocator returned. Snapshotted so FloorMonotonic can
               \* be phrased as a state invariant.
    published  \* SUBSET (1..MaxVersion). Rows in {prefix}_index_watermark.

vars == <<at, seen, pc, attempt, cand, ret, retPubMax, published>>

Max2(x, y) == IF x > y THEN x ELSE y
MaxOf(S)   == CHOOSE x \in S : \A y \in S : y <= x

Inserted    == { c \in ClaimIds : at[c] # 0 }      \* durably accepted rows
Visible(a)  == { c \in Inserted : c \in seen[a] }   \* what a's reads return

\* max_version("capture_version_claims","version")  -- version_allocator.cpp:53
MaxClaimVersion(a) ==
    IF Visible(a) = {} THEN 0 ELSE MaxOf({ at[c] : c \in Visible(a) })
\* max_version("index_watermark","index_version")   -- version_allocator.cpp:54
MaxPublished    == IF published = {} THEN 0 ELSE MaxOf(published)

\* spread -- version_allocator.cpp:57-61. 0 on the first attempt, otherwise
\* uniform in [0, 8*attempt]. We bound it by MaxSpread to close the model.
Spreads(k) == IF k = 0 THEN {0} ELSE 0 .. MaxSpread

TypeOK ==
    /\ at \in [ClaimIds -> 0 .. MaxVersion]
    /\ seen \in [Allocators -> SUBSET ClaimIds]
    /\ pc \in [Allocators -> {"idle","picked","inserted","done","failed",
                              "published","refused"}]
    /\ attempt \in [Allocators -> 0 .. Attempts - 1]
    /\ cand \in [Allocators -> 0 .. MaxVersion]
    /\ ret \in [Allocators -> 0 .. MaxVersion]
    /\ retPubMax \in [Allocators -> 0 .. MaxVersion]
    /\ published \subseteq (1 .. MaxVersion)

Init ==
    /\ at        = [c \in ClaimIds  |-> 0]
    /\ seen      = [a \in Allocators |-> {}]
    /\ pc        = [a \in Allocators |-> "idle"]
    /\ attempt   = [a \in Allocators |-> 0]
    /\ cand      = [a \in Allocators |-> 0]
    /\ ret       = [a \in Allocators |-> 0]
    /\ retPubMax = [a \in Allocators |-> 0]
    /\ published = {}

(***************************************************************************)
(* Pick: the two deciding reads that compute `floor`, plus the choice of   *)
(* jitter.  version_allocator.cpp:53-63.                                   *)
(*                                                                         *)
(* MODELLING NOTE: the code does TWO separate reads (claims, then the      *)
(* watermark); we take them in one atomic step.  Both quantities are       *)
(* monotonically non-decreasing, so splitting them could only yield a      *)
(* floor <= the atomic one -- i.e. a staler floor.  Since every property   *)
(* that can be broken by a stale floor is already broken in this model     *)
(* (see FloorMonotonic), the merge hides nothing we go on to claim.        *)
(***************************************************************************)
Pick(a) ==
    /\ pc[a] = "idle"
    /\ LET fl == Max2(MaxClaimVersion(a), MaxPublished) IN
       \E s \in Spreads(attempt[a]) :
           /\ fl + 1 + s <= MaxVersion       \* STATE-SPACE BOUND, see report
           /\ cand' = [cand EXCEPT ![a] = fl + 1 + s]
    /\ pc' = [pc EXCEPT ![a] = "picked"]
    /\ UNCHANGED <<at, seen, attempt, ret, retPubMax, published>>

(***************************************************************************)
(* Insert: the quorum INSERT of (candidate, claim_id).                     *)
(* version_allocator.cpp:65-71 + quorum_write() at :32-38.                 *)
(***************************************************************************)
Insert(a) ==
    /\ pc[a] = "picked"
    /\ LET c == <<a, attempt[a]>> IN
       /\ at' = [at EXCEPT ![c] = cand[a]]
       /\ seen' = IF Mode = "Linearizable"
                  THEN [b \in Allocators |-> seen[b] \cup {c}]
                  ELSE seen
    /\ pc' = [pc EXCEPT ![a] = "inserted"]
    /\ UNCHANGED <<cand, attempt, ret, retPubMax, published>>

(***************************************************************************)
(* Reveal: an accepted-but-not-yet-visible row becomes readable.  Disabled *)
(* under Linearizable.  This is precisely what                             *)
(* select_sequential_consistency=1 (clickhouse_client.cpp:107) is supposed *)
(* to rule out: a replica answering a read from behind the quorum.         *)
(*                                                                         *)
(* EventuallyConsistent reveals the row to ONE allocator -- two allocators *)
(* can be talking to replicas at different points.  SharedStaleFrontier    *)
(* reveals it to ALL at once -- the store lags, but every reader lags by   *)
(* the same amount.  The difference decides obligation 3; see the report.  *)
(***************************************************************************)
Reveal ==
    /\ Mode # "Linearizable"
    /\ \E c \in Inserted :
         IF Mode = "SharedStaleFrontier"
         THEN /\ \E b \in Allocators : c \notin seen[b]
              /\ seen' = [b \in Allocators |-> seen[b] \cup {c}]
         ELSE \E b \in Allocators :
                /\ c \notin seen[b]
                /\ seen' = [seen EXCEPT ![b] = seen[b] \cup {c}]
    /\ UNCHANGED <<at, pc, attempt, cand, ret, retPubMax, published>>

(***************************************************************************)
(* ReadBack: the deciding read of every claim_id at `candidate`, and the   *)
(* ownership test.  version_allocator.cpp:72-84, and the budget throw at   *)
(* :86-89.                                                                 *)
(***************************************************************************)
ReadBack(a) ==
    /\ pc[a] = "inserted"
    /\ LET me     == <<a, attempt[a]>>
           owners == { c \in Visible(a) : at[c] = cand[a] }
       IN IF owners = {me}
          THEN /\ pc'        = [pc EXCEPT ![a] = "done"]
               /\ ret'       = [ret EXCEPT ![a] = cand[a]]
               /\ retPubMax' = [retPubMax EXCEPT ![a] = MaxPublished]
               /\ UNCHANGED attempt
          ELSE /\ UNCHANGED <<ret, retPubMax>>
               /\ IF attempt[a] = Attempts - 1
                  THEN /\ pc' = [pc EXCEPT ![a] = "failed"]  \* throws
                       /\ UNCHANGED attempt
                  ELSE /\ pc' = [pc EXCEPT ![a] = "idle"]
                       /\ attempt' = [attempt EXCEPT ![a] = attempt[a] + 1]
    /\ UNCHANGED <<at, seen, cand, published>>

(***************************************************************************)
(* Publish: the conditional watermark INSERT the caller runs with the      *)
(* allocated version.  catalog_writer.cpp:587-596 guards it with           *)
(*   coalesce((SELECT max(index_version) FROM index_watermark),0) < V      *)
(* and the read-back at :613-620 turns a refusal into kPublishRace.        *)
(* This action is NOT part of the allocator; it is here so obligation 4    *)
(* (FloorMonotonic) has something to be about.                            *)
(***************************************************************************)
Publish(a) ==
    /\ pc[a] = "done"
    /\ IF ret[a] > MaxPublished
       THEN /\ published' = published \cup {ret[a]}
            /\ pc' = [pc EXCEPT ![a] = "published"]
       ELSE /\ UNCHANGED published
            /\ pc' = [pc EXCEPT ![a] = "refused"]   \* kPublishRace
    /\ UNCHANGED <<at, seen, attempt, cand, ret, retPubMax>>

Next == \/ \E a \in Allocators : Pick(a) \/ Insert(a) \/ ReadBack(a) \/ Publish(a)
        \/ Reveal

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(*                              OBLIGATIONS                                *)
-----------------------------------------------------------------------------

(***************************************************************************)
(* O1  Distinct -- the protocol's ACTUAL guarantee.                        *)
(*     version_allocator.h:5-7: "proceed only as the sole claimant".       *)
(*     EXPECTED: TRUE under Linearizable, FALSE under EventuallyConsistent *)
(*     (see the report for the SharedStaleFrontier surprise).               *)
(***************************************************************************)
Distinct ==
    \A a, b \in Allocators :
        (a # b /\ ret[a] # 0 /\ ret[b] # 0) => ret[a] # ret[b]

(***************************************************************************)
(* O2  SoleRow -- the claim people MISREAD the header as making:           *)
(*     "exactly one claim row stands at a returned version".               *)
(*     EXPECTED: FALSE, even under Linearizable.  A loser's INSERT can     *)
(*     land after the winner's read-back.  This is the invariant           *)
(*     tests/test_native_catalog_lease_live.py:598-607 says is false       *)
(*     ("DURABLY CLAIMED, not solely claimed").                            *)
(***************************************************************************)
SoleRow ==
    \A a \in Allocators :
        ret[a] # 0 => Cardinality({ c \in Inserted : at[c] = ret[a] }) = 1

(***************************************************************************)
(* O3  FloorMonotonic -- no returned version is <= a published            *)
(*     index_version at the moment it is returned.                         *)
(*     EXPECTED: FALSE.  floor is read, not held; the watermark can        *)
(*     overtake it before the read-back.  catalog_writer.cpp:595 is the    *)
(*     guard that actually enforces monotonicity, not the allocator.       *)
(***************************************************************************)
FloorMonotonic ==
    \A a \in Allocators : ret[a] # 0 => ret[a] > retPubMax[a]

(***************************************************************************)
(* O4  NoPublishRefused -- a returned version is always publishable.       *)
(*     EXPECTED: FALSE, and it is the same race as O3 seen downstream:     *)
(*     the conditional INSERT at catalog_writer.cpp:595 refuses, and       *)
(*     :613-620 raises kPublishRace.                                       *)
(***************************************************************************)
NoPublishRefused == \A a \in Allocators : pc[a] # "refused"

(***************************************************************************)
(* O5  NoBudgetExhaustion -- nobody burns the whole attempt budget.        *)
(*     EXPECTED: FALSE.  version_allocator.cpp:86 throws kAllocation.      *)
(***************************************************************************)
NoBudgetExhaustion == \A a \in Allocators : pc[a] # "failed"

(***************************************************************************)
(* VACUITY PROBE -- is the artificial MaxVersion ceiling ever reached?     *)
(* If TLC reports NO violation of this, the ceiling never blocked a Pick   *)
(* and the bound cannot have hidden behaviour.                             *)
(***************************************************************************)
CeilingNeverBinds ==
    \A a \in Allocators :
        pc[a] = "idle" => Max2(MaxClaimVersion(a), MaxPublished) + 1 <= MaxVersion

=============================================================================
