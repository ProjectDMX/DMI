------------------------- MODULE PackPipelineUpload -------------------------
(***************************************************************************)
(* ONE PACK'S UPLOAD under a Cancellation, on main @ 5b3b632:              *)
(*   SpoolUploader::UploadOne   uploader.cpp:103-265  (its attempts, its   *)
(*                              backoff, preflight HEAD/GET, PUT, HEAD,    *)
(*                              Remove, and how the end is booked)         *)
(*   S3Client::ExchangeWith     s3_client.cpp:214-386 (a request's own     *)
(*                              retries and backoff, the progress-callback *)
(*                              abort)                                     *)
(*   Cancellation               cancel.h                                   *)
(*                                                                         *)
(* The PUT is one request (PutSingle); a multipart upload's parts are more *)
(* requests of the same kind.  A request that reaches the server may land  *)
(* its effect and still report a failure (the response is lost or cut).    *)
(* Time: a backoff ends on its timer (the *Timer actions) only when time   *)
(* passes.  Liveness is checked WITHOUT fairness on them, so "a cancel     *)
(* ends the upload" must hold even if no timer ever fires: both backoffs   *)
(* have to wake for the cancel, not merely notice it afterwards.           *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
  MaxAttempts,       \* UploaderConfig::max_attempts
  MaxTries,          \* S3Config::max_attempts (one request's own retries)
  Foreign,           \* TRUE: a DIFFERENT object may already sit at the key
  \* code-shape switches; MAIN in brackets
  UploaderWakes,     \* [TRUE]  SleepBackoff -> Cancellation::SleepFor
  ClientWakes,       \* [TRUE]  Backoff -> Cancellation::SleepFor
  BookLastCut        \* [TRUE]  "if (!cancelled && attempt_cut) cancelled"

VARIABLES
  cancel,            \* the Cancellation (Cancel() or a deadline passed)
  obj,               \* the object at the key: "none" | "ours" | "foreign"
  staged,            \* the .ready file is still in the spool
  attempt,           \* UploadOne's loop index
  pc,                \* UploadOne's step
  next,              \* the step after the request in flight returns OK
  req,               \* the request in flight: "none" | "HEAD" | "GET" | "PUT" | "HEAD2"
  try,               \* ExchangeWith's attempt
  rpc,               \* "send" | "backoff"
  rok,               \* the request's outcome: "ok" | "fail" | "cut"
  attemptCut,        \* attempt_cut
  cancelled,         \* `cancelled`
  result,            \* "none" | "ok" | "fail" | "cancelled"
  sentAfterCancel,   \* history: a request went out after the cancel
  lastReqCut         \* history: the last request of the upload was cut

vars == <<cancel, obj, staged, attempt, pc, next, req, try, rpc, rok,
          attemptCut, cancelled, result, sentAfterCancel, lastReqCut>>

Init ==
  /\ cancel = FALSE
  /\ obj \in (IF Foreign THEN {"none", "foreign"} ELSE {"none"})
  /\ staged = TRUE /\ attempt = 0 /\ pc = "top" /\ next = "none"
  /\ req = "none" /\ try = 0 /\ rpc = "send" /\ rok = "ok"
  /\ attemptCut = FALSE /\ cancelled = FALSE /\ result = "none"
  /\ sentAfterCancel = FALSE /\ lastReqCut = FALSE

Cancel == ~cancel /\ cancel' = TRUE
          /\ UNCHANGED <<obj, staged, attempt, pc, next, req, try, rpc, rok,
                         attemptCut, cancelled, result, sentAfterCancel,
                         lastReqCut>>

-----------------------------------------------------------------------------
(* The uploader's loop (uploader.cpp:117-249)                               *)

\* :117-133.  Past the first attempt the backoff comes first; a cancel
\* ends the retries (the backoff wakes), and none starts after it.
Top ==
  /\ pc = "top" /\ result = "none"
  /\ IF attempt >= MaxAttempts
       THEN /\ pc' = "book" /\ UNCHANGED <<attempt, attemptCut>>
       ELSE IF attempt > 0
              THEN /\ pc' = "ubackoff" /\ UNCHANGED <<attempt, attemptCut>>
              ELSE /\ pc' = "check" /\ UNCHANGED <<attempt, attemptCut>>
  /\ UNCHANGED <<cancel, obj, staged, next, req, try, rpc, rok, cancelled,
                 result, sentAfterCancel, lastReqCut>>

\* SleepBackoff: its timer, or (UploaderWakes) the cancel.
UBackoffTimer ==
  /\ pc = "ubackoff" /\ pc' = "check"
  /\ UNCHANGED <<cancel, obj, staged, attempt, next, req, try, rpc, rok,
                 attemptCut, cancelled, result, sentAfterCancel, lastReqCut>>
UBackoffWake ==
  /\ pc = "ubackoff" /\ cancel /\ UploaderWakes
  /\ cancelled' = TRUE /\ pc' = "book"
  /\ UNCHANGED <<cancel, obj, staged, attempt, next, req, try, rpc, rok,
                 attemptCut, result, sentAfterCancel, lastReqCut>>

Check ==
  /\ pc = "check"
  /\ IF cancel
       THEN cancelled' = TRUE /\ pc' = "book" /\ UNCHANGED <<attemptCut, req, next, try, rpc>>
       ELSE /\ attemptCut' = FALSE                    \* :130-131
            /\ req' = "HEAD" /\ next' = "preflight" /\ try' = 0 /\ rpc' = "send"
            /\ pc' = "req" /\ UNCHANGED cancelled
  /\ UNCHANGED <<cancel, obj, staged, attempt, rok, result, sentAfterCancel,
                 lastReqCut>>

\* What a request returned: ok goes on; a failure records attempt_cut and
\* `continue`s to the next attempt (:139-142, :153-156, :209-213, :229-233).
Returned ==
  /\ pc = "ret"
  /\ IF rok = "ok"
       THEN /\ pc' = next /\ UNCHANGED <<attempt, attemptCut>>
       ELSE /\ attemptCut' = (rok = "cut")
            /\ attempt' = attempt + 1 /\ pc' = "top"
  /\ req' = "none"
  /\ UNCHANGED <<cancel, obj, staged, next, try, rpc, rok, cancelled, result,
                 sentAfterCancel, lastReqCut>>

\* Preflight (:136-197): the object at the key.  Ours: GET it and verify,
\* then Remove.  Foreign: refuse, NOT retryable, never PUT over it.  None:
\* read the staged bytes and PUT.
Preflight ==
  /\ pc = "preflight"
  /\ CASE obj = "ours"    -> /\ req' = "GET" /\ next' = "remove"
                             /\ try' = 0 /\ rpc' = "send" /\ pc' = "req"
                             /\ UNCHANGED result
       [] obj = "foreign" -> /\ result' = "fail" /\ pc' = "end"
                             /\ UNCHANGED <<req, next, try, rpc>>
       [] OTHER           -> /\ req' = "PUT" /\ next' = "posthead"
                             /\ try' = 0 /\ rpc' = "send" /\ pc' = "req"
                             /\ UNCHANGED result
  /\ UNCHANGED <<cancel, obj, staged, attempt, rok, attemptCut, cancelled,
                 sentAfterCancel, lastReqCut>>

PostHead ==
  /\ pc = "posthead"
  /\ req' = "HEAD2" /\ next' = "remove" /\ try' = 0 /\ rpc' = "send"
  /\ pc' = "req"
  /\ UNCHANGED <<cancel, obj, staged, attempt, rok, attemptCut, cancelled,
                 result, sentAfterCancel, lastReqCut>>

\* Remove() after the object is verified (:159, :237): the only exit "ok".
Remove ==
  /\ pc = "remove"
  /\ staged' = FALSE /\ result' = "ok" /\ pc' = "end"
  /\ UNCHANGED <<cancel, obj, attempt, next, req, try, rpc, rok, attemptCut,
                 cancelled, sentAfterCancel, lastReqCut>>

\* :250-264: the attempts are over.  The last attempt cut short by a
\* cancel books a cancel, unless it was already one.
Book ==
  /\ pc = "book"
  /\ result' = IF cancelled \/ (BookLastCut /\ attemptCut)
                 THEN "cancelled" ELSE "fail"
  /\ pc' = "end"
  /\ UNCHANGED <<cancel, obj, staged, attempt, next, req, try, rpc, rok,
                 attemptCut, cancelled, sentAfterCancel, lastReqCut>>

-----------------------------------------------------------------------------
(* One request: ExchangeWith (s3_client.cpp:271-386)                        *)

\* :273-274 nothing goes out once cancelled, the first attempt included.
SendStart ==
  /\ pc = "req" /\ rpc = "send"
  /\ IF cancel
       THEN /\ rok' = "cut" /\ pc' = "ret" /\ lastReqCut' = TRUE
            /\ UNCHANGED <<rpc, sentAfterCancel>>
       ELSE /\ rpc' = "inflight" /\ UNCHANGED <<rok, pc, lastReqCut>>
            /\ UNCHANGED sentAfterCancel
  /\ UNCHANGED <<cancel, obj, staged, attempt, next, req, try, attemptCut,
                 cancelled, result>>

\* The request ends.  It may have landed (a PUT's object exists) whatever
\* it reports.  The progress callback aborts it only once the cancel is set
\* (:96-99, :362); a cut PUT may still have landed (the server finished).
SendEnd ==
  /\ pc = "req" /\ rpc = "inflight"
  /\ \E landed \in BOOLEAN, outcome \in {"ok", "retry", "fail", "abort"}:
       /\ outcome = "abort" => cancel
       /\ obj' = IF req = "PUT" /\ (landed \/ outcome = "ok") THEN "ours"
                 ELSE obj
       /\ CASE outcome = "ok" ->
                 \* HEAD2 (:224-235) fails unless the object is there
                 /\ rok' = IF req = "HEAD2" /\ obj' # "ours" THEN "fail" ELSE "ok"
                 /\ pc' = "ret" /\ lastReqCut' = FALSE /\ rpc' = "send"
                 /\ UNCHANGED try
            [] outcome = "retry" /\ try + 1 < MaxTries ->
                 /\ rpc' = "backoff" /\ UNCHANGED <<rok, pc, try, lastReqCut>>
            [] outcome = "abort" ->
                 /\ rok' = "cut" /\ pc' = "ret" /\ lastReqCut' = TRUE
                 /\ rpc' = "send" /\ UNCHANGED try
            [] OTHER ->
                 /\ rok' = "fail" /\ pc' = "ret" /\ lastReqCut' = FALSE
                 /\ rpc' = "send" /\ UNCHANGED try
  /\ UNCHANGED <<cancel, staged, attempt, next, req, attemptCut, cancelled,
                 result, sentAfterCancel>>

\* :366, :376 Backoff: its timer, or (ClientWakes) the cancel.
CBackoffTimer ==
  /\ pc = "req" /\ rpc = "backoff"
  /\ rpc' = "send" /\ try' = try + 1
  /\ UNCHANGED <<cancel, obj, staged, attempt, pc, next, req, rok, attemptCut,
                 cancelled, result, sentAfterCancel, lastReqCut>>
CBackoffWake ==
  /\ pc = "req" /\ rpc = "backoff" /\ cancel /\ ClientWakes
  /\ rok' = "cut" /\ pc' = "ret" /\ rpc' = "send" /\ lastReqCut' = TRUE
  /\ UNCHANGED <<cancel, obj, staged, attempt, next, req, try, attemptCut,
                 cancelled, result, sentAfterCancel>>

-----------------------------------------------------------------------------

Progress ==
  \/ Top \/ UBackoffWake \/ Check \/ Returned \/ Preflight \/ PostHead
  \/ Remove \/ Book \/ SendStart \/ SendEnd \/ CBackoffWake
Timers == UBackoffTimer \/ CBackoffTimer

Next == Cancel \/ Progress \/ Timers \/ (pc = "end" /\ UNCHANGED vars)

Spec == Init /\ [][Next]_vars
\* Fair progress, NO fairness on the timers: time may stand still.
LiveSpec == Spec /\ WF_vars(Progress) /\ WF_vars(Cancel)

-----------------------------------------------------------------------------

\* The spool copy goes only once the object is ours.
RemoveOnlyVerified == ~staged => obj = "ours"

\* A different object at the key is never overwritten (:178-197).
ForeignKept == [][obj = "foreign" => obj' = "foreign"]_vars

\* "Book an upload as cancelled only when the cancel ended it."
BookedCancelSound == result = "cancelled" => cancel
\* "a cut PUT or part on the last attempt books a cancel"
LastCutBooked == (result # "none" /\ pc = "end" /\ result # "ok"
                  /\ lastReqCut /\ ~(obj = "foreign")) => result = "cancelled"

\* A cancel ends the upload, even if no backoff timer ever fires.
CancelEnds == cancel ~> (pc = "end")

VacOk        == result # "ok"
VacCancelled == result # "cancelled"
VacFail      == result # "fail"
\* A failure stays one when the cancel came while real failures ran out.
VacFailAfterCancel == ~(result = "fail" /\ cancel /\ obj # "foreign")
=============================================================================
