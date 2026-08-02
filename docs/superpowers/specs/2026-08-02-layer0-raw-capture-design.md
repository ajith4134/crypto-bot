# Layer 0 — Raw Market Data Capture

**Date:** 2026-08-02
**Status:** design approved, pending spec review
**Sub-project 1 of 7** in the Autonomous Crypto Trading System (`~/research/DECISIONS.md`)

---

## 1. Context

The system has a settled stack (`~/research/ARCHITECTURE.md` §3c: Python 3.12, NautilusTrader,
Parquet + DuckDB, LightGBM), six research files with 418 catalogued candidate features, and **zero
lines of code**. Prior attempt: `github.com/ajith4134/nse-crypto-bot-final`.

The corpus converged independently, from four different altitudes, on one conclusion:
**build the thing that tells you when you are wrong before the thing that tries to be right.**
Layer 0 is the root of that dependency graph.

**Why this sub-project is first:** it is the only genuinely irreversible item. Validation, features,
backtest and live all consume it, and **market data you did not capture cannot be re-collected at
any price.** Everything else can be retrofitted.

## 2. Scope

Build a **raw capture service**: subscribe to venue websockets, write what arrives verbatim, forever.

### Non-goals — each is a later sub-project

Conflating any of these is how Layer 0 becomes unfinishable.

- No parsing, normalising, or schema-fitting of message bodies
- No Parquet store, no DuckDB, no feature computation
- **No clock-gated access API** — sub-project 2, reads from this archive
- No trading, no signals, no NautilusTrader integration
- No backfill of history — live capture only (backfill is a separate, later decision)

### Governing invariant

> **A written frame is never modified. The bytes on disk are exactly the bytes the venue sent.**
> All derived data — timestamps, sequence numbers, gap records — lives in **sidecar** files.

This is what keeps the archive an option on questions not yet conceived, and what keeps the
exchange-time vs receipt-time distinction (`~/research/IDEAS-STRATEGIC.md` §10) recoverable rather
than baked in wrong.

## 3. Decisions, with rationale

| Decision | Choice | Rationale |
|---|---|---|
| Sequencing | **Raw recorder first**, store/API later | Days of capture beat weeks of architecture. Raw bytes are kept forever, so the store can be rebuilt any number of times |
| Capture scope | **Tiered**: deep core + broad tail | Resolves the tension between "L2 depth is P0" (`ARCHITECTURE.md`) and "the edge is in capacity-constrained corners" (`IDEAS-STRATEGIC.md` §2). Depth is what explodes storage; only a handful of symbols get it |
| Hosting | **This GCE box + GCS offload** | 96 GB local is a ceiling, not a home, for a forever-archive |
| Integrity | **Gap-aware single recorder** | A recorder that silently drops data is worse than none, because it will be trusted |
| Implementation | **Purpose-built async Python** | See §3.1 |

### 3.1 Why not ccxt.pro or NautilusTrader adapters

- **ccxt.pro** normalises messages into its own schema. That **destroys raw fidelity**, which is the
  entire reason for capturing first. You would permanently store ccxt's interpretation with no way
  to recover what the venue actually sent. Also licensed/paid.
- **NautilusTrader adapters** invert a settled decision. `ARCHITECTURE.md` §3c: *"Nautilus must not
  own Layer 0 — the truth layer is ours, and Nautilus consumes from it."* Using its adapters here
  couples the irreplaceable archive to a framework version. Nautilus remains the execution path; it
  does not own capture.

### 3.2 Capture tiers

| Tier | Symbols | Streams |
|---|---|---|
| **Core** | ~4–5 majors, both venues | Full L2 depth diffs + trades + funding + open interest + liquidations |
| **Tail** | As wide as venues stream | Trades + funding + open interest + liquidations. **No depth** |

Tail breadth is a requirement, not a default — see §8 R1.

## 4. Architecture

Six units, one responsibility each.

| Unit | Responsibility |
|---|---|
| `venue_recorder` | One process per venue. Owns subscriptions, reconnect, backoff |
| `raw_writer` | Appends frames verbatim, zstd, hourly rotation, one file per (venue, stream) |
| `capture_ledger` | Records gaps, disconnects, reconnects, queue overflow as first-class events |
| `universe_tracker` | Polls instrument lists; emits listing / delisting / rename / status-change events |
| `archive_offloader` | Checksums completed files, uploads to GCS, prunes local after verified upload |
| `capture_health` | Heartbeat, disk headroom, per-stream staleness, alerting |

**Flow:** websocket frame → bounded in-memory queue → `raw_writer` appends → hourly rotation →
checksum → GCS upload → verify → local prune after retention.

The bounded queue matters: writing must never block socket reads, or backpressure becomes silent
frame loss. **Queue overflow is itself a ledger event**, never a silent drop.

## 5. On-disk format

### 5.1 Verified venue behaviour

Probed live on 2026-08-02 (`scratchpad/probe_venues.py`, 6 frames per stream):

```
binance-spot-depth      frames=6 text=6 binary=0 multiline=0 max_len=1089
binance-futures-depth   frames=6 text=6 binary=0 multiline=0 max_len=1127
hyperliquid l2Book      frames=6 text=6 binary=0 multiline=0 max_len=1610
```

All text, zero binary, **zero embedded newlines** — so NDJSON line alignment is safe.

### 5.2 Two-file layout

```
{stream}_{symbol}_{hourUTC}.ndjson.zst   one venue frame per line, byte-identical to what arrived
{stream}_{symbol}_{hourUTC}.idx.zst      one line per frame, parallel:
                                         {"n":0,"t_recv_ns":…,"t_exch_ms":…,"seq":…,"kind":"data"}
```

Two parallel files rather than an envelope wrapping the payload:

- **Byte-exact** — no base64 (≈33% inflation, worse compression), no JSON re-encoding, no key reordering
- **All derived metadata in the index**, so the raw file is never touched to add a field later
- **Greppable with standard tools**, which matters during an incident
- **Newline guard**: if a payload ever contains a literal newline, escape it and set a flag in the
  index rather than silently corrupting line alignment

### 5.3 Directory layout

```
~/capture/raw/{venue}/{date}/{stream}_{symbol}_{hourUTC}.ndjson.zst + .idx.zst
~/capture/ledger/{venue}/{date}/events.ndjson
~/capture/universe/{venue}/{date}/instruments.ndjson
```

### 5.4 Per-venue sequencing — not uniform

Discovered by probe; designing one mechanism for both venues would have been wrong in both directions.

| Stream | Sequencing | Gap meaning |
|---|---|---|
| Binance spot depth | `U` / `u` (first/final update id) | **Stateful diffs** — gap corrupts the book until REST resync |
| Binance futures depth | `U` / `u` / **`pu`** (previous final id) | Same, plus explicit chain validation |
| Hyperliquid `l2Book` | **no sequence number** | **Stateless snapshots** — gap loses an observation; nothing corrupts |

**Binance futures carries both `E` (event time) and `T` (transaction time)**; spot carries only `E`.
The index records *which* timestamp field it captured per stream rather than assuming a single
"exchange timestamp".

**Control frames** (e.g. Hyperliquid's `subscriptionResponse`) are part of the session record:
**captured**, but flagged `kind:"control"` in the index so downstream does not parse them as data.

### 5.5 Storage projection

Measured frame sizes ~300–1600 bytes at ~10/sec per depth stream ≈ 400 MB/day/stream uncompressed,
~45 MB compressed. Core tier + broad trades-only tail ≈ **1–2 GB/day compressed**. Against 91 GB
free that is 45–90 days of headroom, so **7-day local retention + GCS offload is comfortable.**

## 6. Failure handling

**Principle: never drop data silently, never modify data to "fix" it.** Every anomaly becomes a
ledger event; raw bytes stay untouched even when malformed.

| Failure | Response |
|---|---|
| Websocket disconnect | Exponential backoff. Ledger event with disconnect + reconnect times. Binance depth: mark book invalid, REST snapshot resync, record the resync |
| Sequence-chain break | Ledger event, **severity=corrupting**, resync. Downstream must be able to exclude the window |
| Hyperliquid staleness | Time-based only: no frame in N× expected interval → **severity=observation-loss** |
| Writer queue overflow | Ledger event, **loud alert**. The only path to true data loss |
| Disk pressure | Degradation ladder — see below |
| GCS upload failure | Retry with backoff. **Never prune on a timer — only after checksum-verified upload.** Alert on backlog growth |
| Process crash | Restart writes a `gap_on_restart` event covering the dark window |
| Clock jump / NTP skew | Detect monotonic-vs-wall divergence, flag affected index entries. Timestamps cannot be repaired later |
| Malformed frame | **Written verbatim**, flagged in index, never discarded |
| Symbol delisted mid-stream | `universe_tracker` emits event; recorder unsubscribes cleanly and records why |

**Disk degradation ladder** (`IDEAS-FRONTIER.md` §7): normal → **tail stops, core continues** →
**core depth only** → **halt with loud alert**. Automatic in both directions. Core dies last,
because it is least replaceable.

## 7. Verification

### 7.1 Tests before unattended operation

- **Round-trip byte-exactness** — write frames, read back, assert identical to what arrived
- **Raw/index alignment** — property test: index line *n* always describes raw line *n*, including
  escaped-newline frames
- **Synthetic gap injection** — replay a stream with known dropped sequences; assert the ledger
  records *exactly* those gaps, no more, no fewer
- **Crash safety** — `kill -9` mid-write; assert no torn line breaks alignment on restart
- **Prune safety** — assert prune cannot execute against a file whose upload checksum is unverified

### 7.2 Standing artifact — daily capture integrity report

Per venue/stream: frames captured, sequence continuity, gap count by severity, total dark time,
bytes local vs uploaded, checksum failures.

This makes "is capture healthy?" answerable rather than assumed. Per the guardrail lesson already
recorded in memory: **a green test suite proves logic, not that the thing is firing in production.**

### 7.3 Acceptance criteria

**7 consecutive days unattended**, with every gap in the ledger explained, zero unverified prunes,
and zero silent losses.

## 8. Requirements from the universe-wide scanning design note

Source: `~/research/DESIGN-NOTE-universe-wide-scanning.md`. Both are unrecoverable if missed.

### R1 — The broad tail must be genuinely broad

The scanning idea's value is a **breadth** play (IR ≈ IC × √breadth). Widening the tail is cheap
today and impossible to backfill. **Do not settle for a token 20 symbols.**

### R2 — Point-in-time universe membership must be recorded

Backtesting "watch every symbol" against *today's* symbol list silently conditions on survival —
**survivorship bias entering through the universe definition rather than through prices.** No
purge/embargo scheme catches it.

`universe_tracker` must capture as timestamped first-class events: **listings, delistings, symbol
renames, contract migrations, status changes** (halts, maintenance, settlement changes), plus a
daily full snapshot.

Trivial now. Impossible to reconstruct later — exchanges do not reliably publish historical universe
membership, and third-party reconstructions are exactly the class of unverifiable secondary source
this project has already been burned by once.

## 9. Verified environment facts

Established by direct check on 2026-08-02, not assumed.

| Fact | Status |
|---|---|
| Disk | 96 GB total, **91 GB free** |
| Compute | 12 cores, 29 GB RAM — not a constraint |
| Host | GCE instance `instance-20260801-081737`, project `project-e760fdd7-f8da-46a5-8c4` |
| `gcloud` / `gsutil` | Present (SDK 577.0.0), SA `1095194309870-compute@developer.gserviceaccount.com` active, `cloud-platform` scope |
| `loginctl enable-linger` | **DENIED** — "Access denied" |
| `sudo -n` | **DENIED** — no passwordless sudo |
| `KillUserProcesses` | **false** → detached user processes **survive logout** |
| systemd user unit dir | Writable |

## 10. Blockers and preconditions

### B1 — GCS write access is unproven (blocking for offload only)

```
ERROR: HTTPError 403: ...does not have storage.buckets.list access to the project
```

The SA has `cloud-platform` *scope* but lacks the IAM *role*. **Nuance:** `storage.buckets.list` is
a **project-level** permission; writing objects to one **specific named** bucket requires
`storage.objects.create` on that bucket, which may still be granted independently. Untested — no
bucket name available, and creating a bucket is an outward billable action not taken unprompted.

**Resolution — either:**
1. Grant the SA `roles/storage.objectAdmin` on a dedicated bucket, **or**
2. Provide a bucket name so a real write test can run

**Gating rule:** the recorder ships and runs **local-only from day one** — capture starts
immediately, nothing irreplaceable is lost — and offload is switched on the moment the write test
passes. No code depends on GCS until proven.

### B2 — Reboot persistence unresolved (non-blocking, needs mitigation)

`KillUserProcesses=false` means the recorder survives **logout**. It does **not** survive a VM
**reboot**, because linger cannot be enabled and there is no root systemd access.

Mitigations to evaluate during planning: user `@reboot` crontab (untested), an external watchdog, or
obtaining linger from a project administrator. **Until resolved, a reboot means a silent capture
outage** — so `capture_health` must alert on absence, not merely on error.

## 11. Open questions for planning

1. Exact core symbol list (4–5 majors) and tail universe definition per venue
2. Local retention window — 7 days assumed, confirm against measured daily volume
3. Alert channel for `capture_health` — no notification path exists yet
4. Whether to capture Binance spot, futures, or both for the core tier
