------------------------------ MODULE PackRank ------------------------------
(***************************************************************************)
(* How the native reader picks the pack a capture resolves to, at a       *)
(* pinned watermark, while indexer passes publish, replay and fail, and   *)
(* ClickHouse merges descriptor rows in the background (#161, d3fe7c2).   *)
(*                                                                         *)
(* SOURCE OF TRUTH (main @ 5b3b632):                                       *)
(*                                                                         *)
(*   native/csrc/catalog/reader.cpp                                        *)
(*     :20-21   kSortKey: a capture's identity, what the read groups on.   *)
(*     :24-32   kProjection: what a read returns.  index_version is NOT    *)
(*              in it -- it "orders rather than describes".                *)
(*     :35-41   kResolutionOrder =                                         *)
(*              (member_version, store_id, pack_id, index_version).        *)
(*     :481-497 snapshot(): capture_raw INNER JOIN, on (store_id,          *)
(*              pack_id), min(index_version) AS member_version over the    *)
(*              manifest rows at or below the pin whose (index_version,    *)
(*              publish_id) is a watermark row at or below the pin.  The   *)
(*              descriptor rows themselves are NOT bounded by the pin.     *)
(*     :499-520 projection(): ONE argMax(tuple(...), kResolutionOrder).    *)
(*   native/csrc/catalog/schema.cpp                                        *)
(*     :27-29,498-512  capture_raw is ReplacingMergeTree(index_version)    *)
(*              ORDER BY (identity..., store_id, pack_id): a merge keeps,  *)
(*              per (capture, pack), the row with the greatest version.    *)
(*     :530-540 index_watermark, snapshot_manifest: plain MergeTree.       *)
(*   native/csrc/catalog/indexer.cpp:276-444  commit(): descriptor rows    *)
(*              are written at the allocated version BEFORE the publish;   *)
(*              a lost version race reallocates and rewrites them; a      *)
(*              pass that replays an already-published pack (a crash      *)
(*              before commit_packs, an unknown-outcome publish that       *)
(*              landed, a rebuild) rewrites its rows at a fresh version.   *)
(*   native/csrc/catalog/pack_index.cpp:381-383  a pack carrying one       *)
(*              capture id twice is refused, so within one pack a capture  *)
(*              has one record.                                            *)
(*   native/csrc/sink/object_key.cpp:66-85  the object key is a function   *)
(*              of the pack's own fields, so a replay's descriptor rows    *)
(*              are byte-identical to the first pass's except for          *)
(*              index_version (ReplayIdentical below).                     *)
(*   docs/capture-storage-design.md "The ranking version";                 *)
(*   docs/catalog-descriptor-key.md :87-106.                               *)
(*                                                                         *)
(* ABSTRACTIONS                                                            *)
(*  - store_id is folded into the pack id: (store_id, pack_id) is one      *)
(*    totally ordered pack identity here (Packs are naturals).             *)
(*  - Versions are allocated from a counter: VersionAllocator.tla is what  *)
(*    shows they are distinct.  A publish's watermark row lands only above *)
(*    every landed one (the barrier in the fenced watermark INSERT); that  *)
(*    is PublisherLease.tla's WatermarkMonotonic, and OrderedWatermark =   *)
(*    FALSE drops it to show what this model leans on.                     *)
(*  - A publish is its manifest rows then its watermark row.  Any stage    *)
(*    may fail and leave what it wrote; a failed publish's manifest rows   *)
(*    are orphans, paired with no watermark row.                           *)
(*  - A read is one snapshot of the tables; the two-query page shape and   *)
(*    cursors are not modelled.                                            *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
  Packs,            \* pack identities, naturals: the (store_id, pack_id) order
  Caps,             \* capture identities (kSortKey)
  Passes,           \* indexer passes that may run concurrently
  MaxVer,           \* version bound (model bound, not protocol)
  Ranking,          \* "first"      : (min member version, pack, iv)  -- 5b3b632
                    \* "newest"     : (max member version, pack, iv)  -- #156
                    \* "rowversion" : (index_version, pack)          -- 99ee4ae
  ReplayIdentical,  \* TRUE: a replay rewrites byte-identical rows
  OrderedWatermark  \* TRUE: watermark rows land in version order

VARIABLES
  raw,     \* capture_raw rows [cap, pack, iv]
  man,     \* snapshot_manifest rows [v, pid, pack]
  wm,      \* index_watermark rows [v, pid]
  ver,     \* the last version allocated
  pass,    \* per pass: [st, v, S]
  pins,    \* [1..MaxVer -> {} or {result}]: the first read at each pin
  replays, merges  \* vacuity witnesses

vars == <<raw, man, wm, ver, pass, pins, replays, merges>>

Versions == 1..MaxVer
SetMax(S) == IF S = {} THEN 0 ELSE CHOOSE x \in S : \A y \in S : y <= x
SetMin(S) == CHOOSE x \in S : \A y \in S : x <= y

(* A row's content: what kProjection returns.  index_version is not in it,
   so a replay's row is the same content unless ReplayIdentical is FALSE. *)
Content(r) == IF ReplayIdentical THEN <<r.cap, r.pack>>
                                 ELSE <<r.cap, r.pack, r.iv>>

(* The publish_id of pass p's publish at version v: unique per (p, v). *)
Pid(p, v) == <<p, v>>

----------------------------------------------------------------------------
(* THE READ -- reader.cpp:481-520 *)

Paired(W) == {m \in man : m.v <= W /\
                \E w \in wm : w.v = m.v /\ w.pid = m.pid /\ w.v <= W}
MemberVers(W, k) == {m.v : m \in {q \in Paired(W) : q.pack = k}}
IsMember(W, k) == MemberVers(W, k) # {}

Rank(W, r) ==
  CASE Ranking = "first"      -> <<SetMin(MemberVers(W, r.pack)), r.pack, r.iv>>
    [] Ranking = "newest"     -> <<SetMax(MemberVers(W, r.pack)), r.pack, r.iv>>
    [] Ranking = "rowversion" -> <<r.iv, r.pack, 0>>

LexLess(a, b) ==
  \/ a[1] < b[1]
  \/ a[1] = b[1] /\ a[2] < b[2]
  \/ a[1] = b[1] /\ a[2] = b[2] /\ a[3] < b[3]

Cands(W, c) == {r \in raw : r.cap = c /\ IsMember(W, r.pack)}
Top(W, c) == {r \in Cands(W, c) :
                \A q \in Cands(W, c) : ~LexLess(Rank(W, r), Rank(W, q))}
(* argMax's answer: the contents of the top-ranked rows.  More than one
   content means argMax picks between them by read order -- the flip the
   tuple key exists to rule out. *)
Resolve(W) == [c \in Caps |-> {Content(r) : r \in Top(W, c)}]

----------------------------------------------------------------------------
Init ==
  /\ raw = {} /\ man = {} /\ wm = {} /\ ver = 0
  /\ pass = [p \in Passes |-> [st |-> "idle", v |-> 0, S |-> {}]]
  /\ pins = [W \in Versions |-> {}]
  /\ replays = FALSE /\ merges = FALSE

(* allocate_version, then the pass's packs: new ones or replays.  A pack is
   a replay when an earlier publish already named it (a crash before
   commit_packs, an outcome-unknown publish that landed, a rebuild). *)
Begin(p) ==
  /\ pass[p].st = "idle"
  /\ ver < MaxVer
  /\ \E S \in (SUBSET Packs) \ {{}} :
       /\ pass' = [pass EXCEPT ![p] = [st |-> "rows", v |-> ver + 1, S |-> S]]
       /\ replays' = (replays \/ \E m \in man : m.pack \in S)
  /\ ver' = ver + 1
  /\ UNCHANGED <<raw, man, wm, pins, merges>>

(* The descriptor rows, at the pass's version, before any publish
   (indexer.cpp commit()).  Every pack holds every capture: the case where
   packs compete for one capture. *)
WriteRows(p) ==
  /\ pass[p].st = "rows"
  /\ raw' = raw \cup {[cap |-> c, pack |-> k, iv |-> pass[p].v] :
                        c \in Caps, k \in pass[p].S}
  /\ pass' = [pass EXCEPT ![p].st = "manifest"]
  /\ UNCHANGED <<man, wm, ver, pins, replays, merges>>

WriteManifest(p) ==
  /\ pass[p].st = "manifest"
  /\ man' = man \cup {[v |-> pass[p].v, pid |-> Pid(p, pass[p].v), pack |-> k] :
                        k \in pass[p].S}
  /\ pass' = [pass EXCEPT ![p].st = "watermark"]
  /\ UNCHANGED <<raw, wm, ver, pins, replays, merges>>

(* The fenced watermark INSERT with its version barrier.  A publish that
   loses the barrier is a publish race: the indexer reallocates and tries
   again, which here is a fresh pass. *)
WriteWatermark(p) ==
  /\ pass[p].st = "watermark"
  /\ OrderedWatermark => pass[p].v > SetMax({w.v : w \in wm})
  /\ wm' = wm \cup {[v |-> pass[p].v, pid |-> Pid(p, pass[p].v)]}
  /\ pass' = [pass EXCEPT ![p] = [st |-> "idle", v |-> 0, S |-> {}]]
  /\ UNCHANGED <<raw, man, ver, pins, replays, merges>>

(* Any stage can fail -- a refusal, a fence, a crash -- leaving its rows. *)
Fail(p) ==
  /\ pass[p].st # "idle"
  /\ pass' = [pass EXCEPT ![p] = [st |-> "idle", v |-> 0, S |-> {}]]
  /\ UNCHANGED <<raw, man, wm, ver, pins, replays, merges>>

(* A background merge: two rows of one (capture, pack) -- one sort key in
   capture_raw -- collapse to the one with the greater version. *)
Merge ==
  \E r1, r2 \in raw :
    /\ r1.cap = r2.cap /\ r1.pack = r2.pack /\ r1.iv < r2.iv
    /\ raw' = raw \ {r1}
    /\ merges' = TRUE
    /\ UNCHANGED <<man, wm, ver, pass, pins, replays>>

(* A read pinned at a published watermark.  The first read at each pin is
   recorded; PinnedReadStable compares every later state against it. *)
Pin ==
  \E w \in wm :
    /\ pins[w.v] = {}
    /\ pins' = [pins EXCEPT ![w.v] = {Resolve(w.v)}]
    /\ UNCHANGED <<raw, man, wm, ver, pass, replays, merges>>

Next ==
  \/ \E p \in Passes :
       Begin(p) \/ WriteRows(p) \/ WriteManifest(p) \/ WriteWatermark(p)
       \/ Fail(p)
  \/ Merge \/ Pin

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* OBLIGATIONS *)

TypeOK == /\ ver \in 0..MaxVer
          /\ \A r \in raw : r.iv \in Versions

(* 1. A read pinned at W returns what it returned the first time, through
   every later publish, replay, failed pass and merge. *)
PinnedReadStable ==
  \A W \in Versions : pins[W] # {} => pins[W] = {Resolve(W)}

(* 2. At every published watermark each capture resolves to ONE content. *)
UniqueWinner ==
  \A w \in wm : \A c \in Caps : Cardinality(Resolve(w.v)[c]) <= 1

(* 3. Supersession: at every published watermark a capture resolves to the
   member pack whose FIRST publish is the latest (ties by pack identity).
   A replay adds nothing to the catalog, so it must not promote a pack. *)
FirstPub(W, k) == SetMin(MemberVers(W, k))
Newest(W, c) ==
  CHOOSE k \in {q \in Packs : IsMember(W, q)} :
    \A q \in {q2 \in Packs : IsMember(W, q2)} :
      q # k => (FirstPub(W, q) < FirstPub(W, k) \/
                (FirstPub(W, q) = FirstPub(W, k) /\ q < k))
LatestPackWins ==
  \A w \in wm : \A c \in Caps :
    (\E k \in Packs : IsMember(w.v, k)) =>
      \A r \in Top(w.v, c) : r.pack = Newest(w.v, c)

AllPackRank == PinnedReadStable /\ UniqueWinner /\ LatestPackWins

(* VACUITY GUARDS -- each wanted refuted. *)
NeverReplayed      == ~replays
NeverMerged        == ~merges
(* a pin recorded while two member packs compete for one capture *)
NeverPinnedContest ==
  ~\E w \in wm : pins[w.v] # {} /\
     Cardinality({k \in Packs : IsMember(w.v, k)}) >= 2
(* a replay published above a pin taken while its pack was already a member *)
NeverReplayAbovePin ==
  ~\E W \in Versions, k \in Packs :
     /\ pins[W] # {} /\ IsMember(W, k)
     /\ \E m \in Paired(MaxVer) : m.pack = k /\ m.v > W
=============================================================================
