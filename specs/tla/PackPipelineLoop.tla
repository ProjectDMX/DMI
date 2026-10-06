-------------------------- MODULE PackPipelineLoop --------------------------
(***************************************************************************)
(* WHO RUNS A CYCLE, AND WHEN: the background loop, flush(), and stop()    *)
(* around cycle_mutex_, on main @ 5b3b632.                                 *)
(*                                                                         *)
(*   storage_service.cpp                                                   *)
(*     :490-550  loop(): wait_for(wait_ns | stop | kick); take the cycle   *)
(*               mutex; exit on stop; "wait out the rest of the interval   *)
(*               first, unless a fresh lease asked for a cycle now --      *)
(*               once" (:523-538, waited_again); run_cycle                 *)
(*     :424-478  flush(): try_lock_until(deadline), stop -> false, a       *)
(*               cycle, drained -> true, deadline -> false, sleep, again   *)
(*     :372-413  stop(): stop_requested_, cancel, join the loop            *)
(*   engine.py:944-970  _retire_capture_storage: storage.flush(), then     *)
(*               storage.stop() at once ("Run no loop cycle between a      *)
(*               flush and the stop() after it", #162)                     *)
(*                                                                         *)
(* Time: Elapse makes the last cycle's end "old" (since_ns >= wait_ns) and *)
(* lets the loop's wait time out.  Between flush() returning and stop()    *)
(* being called no time passes: stop() follows at once.                    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
  MaxFlushes,        \* flush() calls before close()'s
  AllowKick,         \* a fresh lease may kick the loop
  WaitAgainOnce      \* [MAIN TRUE] the loop defers to a recent cycle once;
                     \* FALSE: every time

VARIABLES
  lp,          \* loop: "wait" | "lock" | "cycle" | "exit"
  kicked,      \* the wake that took the loop to "lock" was a kick
  waitedAgain, \* waited_again
  kick,        \* kick_
  mutex,       \* cycle_mutex_: "none" | "loop" | "flush"
  recent,      \* steady_ns() - last_cycle_end_ns_ < wait_ns
  cycled,      \* last_cycle_end_ns_ != 0
  fl,          \* flush(): "idle" | "lock" | "cycle" | "sleep"
  flushes,     \* flush() calls made
  cl,          \* the caller: "run" | "flush" | "closeflush" | "between" |
               \*             "stopping" | "stopped"
  stopReq,
  betweenCycle,\* history: a loop cycle started in "between"
  loopCycles   \* history: loop cycles started (capped at 2)

vars == <<lp, kicked, waitedAgain, kick, mutex, recent, cycled, fl, flushes,
          cl, stopReq, betweenCycle, loopCycles>>

Init ==
  /\ lp = "wait" /\ kicked = FALSE /\ waitedAgain = FALSE /\ kick = FALSE
  /\ mutex = "none" /\ recent = FALSE /\ cycled = FALSE
  /\ fl = "idle" /\ flushes = 0 /\ cl = "run" /\ stopReq = FALSE
  /\ betweenCycle = FALSE /\ loopCycles = 0

Elapse ==
  /\ cl # "between" /\ recent
  /\ recent' = FALSE
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, mutex, cycled, fl, flushes,
                 cl, stopReq, betweenCycle, loopCycles>>

Kick ==
  /\ AllowKick /\ ~kick /\ ~stopReq
  /\ kick' = TRUE
  /\ UNCHANGED <<lp, kicked, waitedAgain, mutex, recent, cycled, fl, flushes,
                 cl, stopReq, betweenCycle, loopCycles>>

\* :496-502.  A timeout needs time to pass, so not in "between".
LoopWake ==
  /\ lp = "wait"
  /\ IF stopReq THEN lp' = "exit" /\ UNCHANGED <<kicked, kick>>
     ELSE /\ (kick \/ cl # "between")
          /\ lp' = "lock" /\ kicked' = kick /\ kick' = FALSE
  /\ UNCHANGED <<waitedAgain, mutex, recent, cycled, fl, flushes, cl, stopReq,
                 betweenCycle, loopCycles>>

\* :508-540.
LoopLock ==
  /\ lp = "lock" /\ mutex = "none"
  /\ IF stopReq
       THEN lp' = "exit" /\ UNCHANGED <<mutex, waitedAgain, betweenCycle, loopCycles>>
       ELSE IF ~kicked /\ (~waitedAgain \/ ~WaitAgainOnce) /\ cycled /\ recent
              THEN /\ waitedAgain' = TRUE /\ lp' = "wait"
                   /\ UNCHANGED <<mutex, betweenCycle, loopCycles>>
              ELSE /\ waitedAgain' = FALSE /\ lp' = "cycle" /\ mutex' = "loop"
                   /\ betweenCycle' = (betweenCycle \/ cl = "between")
                   /\ loopCycles' = IF loopCycles < 2 THEN loopCycles + 1 ELSE 2
  /\ UNCHANGED <<kicked, kick, recent, cycled, fl, flushes, cl, stopReq>>

LoopCycleEnd ==
  /\ lp = "cycle"
  /\ lp' = "wait" /\ mutex' = "none" /\ recent' = TRUE /\ cycled' = TRUE
  /\ UNCHANGED <<kicked, waitedAgain, kick, fl, flushes, cl, stopReq,
                 betweenCycle, loopCycles>>

\* The caller: flush_and_wait() while capturing, then close(): its flush,
\* then stop() at once.
CallFlush ==
  /\ cl = "run" /\ fl = "idle" /\ flushes < MaxFlushes
  /\ cl' = "flush" /\ fl' = "lock" /\ flushes' = flushes + 1
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, mutex, recent, cycled, stopReq,
                 betweenCycle, loopCycles>>
CallClose ==
  /\ cl = "run" /\ fl = "idle"
  /\ cl' = "closeflush" /\ fl' = "lock"
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, mutex, recent, cycled, flushes,
                 stopReq, betweenCycle, loopCycles>>

FlushReturns ==
  /\ fl' = "idle"
  /\ cl' = IF cl = "closeflush" THEN "between" ELSE "run"

\* :449: the deadline may pass while the loop holds the mutex.
FlushLock ==
  /\ fl = "lock"
  /\ \/ /\ mutex = "none"
        /\ IF stopReq THEN FlushReturns /\ UNCHANGED mutex
           ELSE fl' = "cycle" /\ mutex' = "flush" /\ UNCHANGED cl
     \/ /\ mutex # "none" /\ FlushReturns /\ UNCHANGED mutex  \* timed out
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, recent, cycled, flushes,
                 stopReq, betweenCycle, loopCycles>>

\* The cycle ends; drained or out of time returns, else sleep and again.
FlushCycleEnd ==
  /\ fl = "cycle"
  /\ mutex' = "none" /\ recent' = TRUE /\ cycled' = TRUE
  /\ \/ FlushReturns
     \/ fl' = "sleep" /\ UNCHANGED cl
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, flushes, stopReq,
                 betweenCycle, loopCycles>>

FlushSleep ==
  /\ fl = "sleep" /\ fl' = "lock"
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, mutex, recent, cycled, flushes,
                 cl, stopReq, betweenCycle, loopCycles>>

Stop ==
  /\ cl = "between"
  /\ stopReq' = TRUE /\ cl' = "stopping"
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, mutex, recent, cycled, fl,
                 flushes, betweenCycle, loopCycles>>

Joined ==
  /\ cl = "stopping" /\ lp = "exit"
  /\ cl' = "stopped"
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, mutex, recent, cycled, fl,
                 flushes, stopReq, betweenCycle, loopCycles>>

Next ==
  \/ Elapse \/ Kick \/ LoopWake \/ LoopLock \/ LoopCycleEnd
  \/ CallFlush \/ CallClose \/ FlushLock \/ FlushCycleEnd \/ FlushSleep
  \/ Stop \/ Joined

Spec == Init /\ [][Next]_vars

\* For the liveness check: flushes that never stop coming (cycle after
\* cycle, as a tight flush_and_wait loop makes), and a fair loop.
EndlessFlushes ==
  /\ cl = "run" /\ fl = "idle"
  /\ cl' = "flush" /\ fl' = "lock"
  /\ UNCHANGED <<lp, kicked, waitedAgain, kick, mutex, recent, cycled, flushes,
                 stopReq, betweenCycle, loopCycles>>
LoopActs == LoopWake \/ LoopLock \/ LoopCycleEnd
FlushActs == FlushLock \/ FlushCycleEnd \/ FlushSleep
LiveNext ==
  \/ Elapse \/ LoopActs \/ FlushActs \/ EndlessFlushes
\* No fairness on Elapse: the flushes come faster than the poll interval,
\* so the last cycle may always be recent.  A thread waiting for the
\* mutex gets it when it is free often enough (SF).
LiveSpec == /\ Init /\ [][LiveNext]_vars
            /\ SF_vars(LoopActs) /\ WF_vars(FlushActs) /\ WF_vars(EndlessFlushes)

-----------------------------------------------------------------------------

\* #162: no loop cycle between close()'s flush and its stop().
NoLoopCycleBeforeStop == ~betweenCycle

\* The loop's own work (the reconcile, adoption) is not starved by
\* flushes that keep coming.
LoopNotStarved == []<>(lp = "cycle")

VacBetween == cl # "between"
VacWaitedAgain == ~(waitedAgain /\ lp = "wait" /\ fl = "cycle")
=============================================================================
