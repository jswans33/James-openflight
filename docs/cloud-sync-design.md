# OpenFlight Cloud Sync — Client-Side Design (Proposal)

Status: **draft / not implemented**. Design for the two pieces of the planned
cloud service that live in this (public) repo: the ingest API contract the Pi
speaks, and the uploader that speaks it. The service itself lives in a
separate private repo; this document is the interface between them.

Design priorities, in order:

1. **The contract is forever.** Fielded Pis update rarely. The wire format
   must tolerate old clients indefinitely and evolve additively.
2. **Never upload raw ADC/I-Q.** Filtered session summaries only — keeps
   storage costs sane, uploads fast on bad wifi, and is the
   privacy-friendly default.
3. **No user babysitting.** Sessions upload themselves when connectivity
   exists; "no wifi at the range" is a non-event, not a manual step.

---

## 1. Ingest API contract (v1)

All endpoints under `https://<cloud-host>/v1/`, TLS only. Authentication is
an opaque per-device bearer token (`of_device_...` prefix) — revocable from
the web app, scoped to one device.

### Device linking (how a Pi gets its token)

Simplified RFC 8628 device-code flow — the user never copies a long token
onto the Pi:

```
Pi                                          Cloud
──                                          ─────
POST /v1/device-link/start
  {device_name, client_version}      ─────▶
                                     ◀───── {link_code: "ABCD-1234",
                                             poll_token, interval_s, expires_s}

  (Pi prints: "Go to cloud.openflight.example/link
   and enter code ABCD-1234")

POST /v1/device-link/poll
  {poll_token}            (repeat)   ─────▶
                                     ◀───── {status: "pending"}
                                     ◀───── {status: "linked",
                                             device_token, device_id}
```

The user enters the code on the website while signed in. Token is stored at
`~/.config/openflight/cloud.json`, mode `0600`, never logged.

### Session upload

```
PUT /v1/sessions/{session_id}
  Authorization: Bearer <device_token>
  Content-Type: application/x-ndjson
  Content-Encoding: gzip

  <filtered session JSONL, gzipped>
```

- **`session_id` is the `session_uuid` embedded in the `session_start`
  entry** (a UUID4 written at session creation since format_version 1 /
  app 0.2.0 — survives file renames and copies). For older sessions that
  predate the field, the uploader falls back to a deterministic UUIDv5 of
  `(device_id, session filename)`. Either way the auto-push and the manual
  push script can both submit the same session and the server dedupes for
  free; PUT semantics make retries safe.
- The client prepends one manifest line to the body:

  ```json
  {"type": "upload_manifest", "format_version": 1, "client_version": "0.2.0",
   "device_id": "...", "filtered": true, "kept_entry_types": ["session_start", ...]}
  ```

**Responses:**

| code | meaning | client behavior |
|---|---|---|
| 201 | accepted (`{session_id, shot_count}`) | mark pushed |
| 200 | duplicate — already stored | mark pushed |
| 401 | token invalid/revoked | stop, flag "needs re-link" |
| 402 | quota/entitlement exceeded | park, retry daily |
| 413 | body too large (cap ~20 MB gzipped) | log error, park file |
| 422 | unparseable (`{reason}`) | log error, park file |
| 429 | rate limited (`Retry-After`) | back off |
| 5xx | server trouble | retry with backoff |

`GET /v1/health` → 200, used by the uploader to short-circuit when offline.

### Evolution rules

- The server must **accept and store unknown entry types** (skip parsing,
  don't reject) — old and new clients coexist for years.
- Additive changes only within `/v1`; breaking changes get `/v2` and `/v1`
  keeps working.

---

## 2. Client-side filtering (the raw-ADC strip)

Filtering uses an **allowlist**, not a blocklist — a future heavy entry type
added to the session logger can never leak to the cloud by accident:

```
keep:  session_start, session_end, shot_detected, trigger_event, session_error
drop:  rolling_buffer_capture, iq_blocks, iq_reading, reading_accepted,
       and anything not on the keep list
guard: any kept line > 32 KB is dropped and counted (belt and suspenders)
```

`shot_detected` carries everything the insights product needs (speeds, spin
+ quality/confidence, angles, carry, K-LD7 diagnostics). `trigger_event` and
`session_error` are small and power reliability insights. Raw I/Q and RADC
stay on the Pi where they belong — they remain available locally for the
offline analysis workflows.

A typical filtered session is **tens of KB gzipped** vs tens of MB raw.

---

## 3. The uploader (spool-and-retry)

Lives in the public repo. Three entry points, one mechanism:

```
openflight-cloud link              # device-link flow (one time)
openflight-cloud push [--dry-run]  # filter + upload anything unpushed
openflight-cloud status            # linked? queued? parked? last error?
```

**Mechanism:**

- The session directory itself is the queue. A session counts as "pushed"
  when a sidecar marker (`<session>.jsonl.pushed`) exists — originals are
  never moved or modified, state survives crashes, and no database is
  involved.
- A **systemd timer** (every ~10 min) runs `push`; the server process also
  fires a non-blocking `push` on session end. The timer makes wifi outages
  self-healing; the hook makes the happy path fast. Neither can ever delay
  shot processing.
- Per-file attempt counter (in the sidecar); after ~20 failures the file is
  parked (`.parked`) and reported by `status` instead of retried forever.
- `--dry-run` prints exactly which entry lines would upload — the privacy
  answer to "what are you sending?"

**Config** (`~/.config/openflight/cloud.json`):

```json
{"endpoint": "https://cloud.openflight.example",
 "device_token": "of_device_...", "device_id": "...", "enabled": true}
```

Uploading is **opt-in**: nothing leaves the Pi until the user runs
`openflight-cloud link`. Later, the interactive `setup.sh` can offer linking
as an optional step.

---

## 4. Server-side checklist (private repo, for reference)

Not designed here, but the contract above implies: the four endpoints;
dedupe on `session_id`; entitlement check at ingest (quota → 402);
blob → object storage; parse kept entries → per-shot rows in Postgres;
device management UI (list/revoke); link-code UI.

## Open questions

- Tier gating: does the free tier get full history or last-N sessions?
  (Affects only server; contract unchanged.)
- Should `iq_reading` summaries (SNR stats, no raw data) join the allowlist
  later for radar-health insights? Cheap to add — allowlist makes it an
  explicit decision.
- AGPL hygiene: the uploader (public repo) is AGPL like everything here;
  the private service must not import code from this repo unless that code
  is dual-licensed or contributor-cleared. Keep the boundary at the wire
  contract.
