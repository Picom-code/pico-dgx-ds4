// Package metrics reads the front-end's /metrics counters and turns them into the 3 readings the
// operator asked for: how many streams are in progress, whether the line is encoding or decoding, and
// the encode and decode rates over a 0.1 s, a 0.5 s and a 1 s window.
//
// A rate is computed from 2 readings of the same counter, never from 1 sample: the counter is monotonic
// since the server started, so the difference between 2 readings divided by the time between them is
// the rate over exactly that window. The package keeps the last second of readings, so a reader that
// asks every 100 ms can answer all 3 windows from the readings it already holds.
package metrics

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Windows are the 3 windows the rates are reported over, shortest first.
var Windows = []time.Duration{
	100 * time.Millisecond,
	500 * time.Millisecond,
	1 * time.Second,
}

// Reading is one sample of the front-end's counters, with the time it was taken.
type Reading struct {
	At            time.Time
	Phase         string
	Streams       int64
	StreamsEnc    int64
	GateInFlight  int64
	GateWaiting   int64
	GatePassed    int64
	GateRefused   int64
	EncodeTokens  int64
	DecodeTokens  int64
	ContextTokens int64
	MaxContext    int64
	EngineUp      bool
}

// Rate is the tokens per second of one counter over one window.
type Rate struct {
	WindowS   float64 `json:"window_s"`
	EncodeTPS float64 `json:"encode_tps"`
	DecodeTPS float64 `json:"decode_tps"`
	// Samples is how many readings the window spans; 0 means the window is not covered yet and the
	// rates are 0 rather than a guess.
	Samples int `json:"samples"`
}

// Snapshot is what the runner reports: the counters now, and the rates over each window.
type Snapshot struct {
	At            time.Time `json:"at"`
	Phase         string    `json:"phase"`
	Streams       int64     `json:"streams"`
	StreamsEnc    int64     `json:"streams_encoding"`
	GateInFlight  int64     `json:"gate_in_flight"`
	GateWaiting   int64     `json:"gate_waiting"`
	EncodeTokens  int64     `json:"encode_tokens"`
	DecodeTokens  int64     `json:"decode_tokens"`
	ContextTokens int64     `json:"context_tokens"`
	MaxContext    int64     `json:"max_context"`
	ContextPct    float64   `json:"context_pct"`
	EngineUp      bool      `json:"engine_connected"`
	Rates         []Rate    `json:"rates"`
	Err           string    `json:"error,omitempty"`
}

// The wire shape of the front-end's /metrics answer.
type wire struct {
	Phase         string `json:"phase"`
	Streams       int64  `json:"streams"`
	StreamsEnc    int64  `json:"streams_encoding"`
	GateInFlight  int64  `json:"gate_in_flight"`
	GateWaiting   int64  `json:"gate_waiting"`
	GatePassed    int64  `json:"gate_passed"`
	GateRefused   int64  `json:"gate_refused"`
	EncodeTokens  int64  `json:"encode_tokens"`
	DecodeTokens  int64  `json:"decode_tokens"`
	ContextTokens int64  `json:"context_tokens"`
	MaxContext    int64  `json:"max_context"`
	EngineUp      bool   `json:"engine_connected"`
}

// Tracker keeps the readings of one front-end address, so that a rate can be computed over a window
// that reaches back before the last request.
type Tracker struct {
	mu       sync.Mutex
	addr     string
	port     string
	readings []Reading
	// keep is how far back the readings are kept: the longest window plus one poll interval.
	keep time.Duration
}

// NewTracker returns a tracker for one front-end address and port.
func NewTracker(addr string, port string) *Tracker {
	return &Tracker{addr: addr, port: port, keep: Windows[len(Windows)-1] + 200*time.Millisecond}
}

// Read fetches /metrics once and returns the reading. It does not record the reading.
func (t *Tracker) Read() (Reading, error) {
	client := &http.Client{Timeout: 3 * time.Second}
	response, err := client.Get("http://" + t.addr + ":" + t.port + "/metrics")
	if err != nil {
		return Reading{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return Reading{}, fmt.Errorf("/metrics answered %d", response.StatusCode)
	}
	var body wire
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		return Reading{}, fmt.Errorf("/metrics is not JSON: %w", err)
	}
	return Reading{
		At:            time.Now(),
		Phase:         body.Phase,
		Streams:       body.Streams,
		StreamsEnc:    body.StreamsEnc,
		GateInFlight:  body.GateInFlight,
		GateWaiting:   body.GateWaiting,
		GatePassed:    body.GatePassed,
		GateRefused:   body.GateRefused,
		EncodeTokens:  body.EncodeTokens,
		DecodeTokens:  body.DecodeTokens,
		ContextTokens: body.ContextTokens,
		MaxContext:    body.MaxContext,
		EngineUp:      body.EngineUp,
	}, nil
}

// Poll reads /metrics once and records the reading, dropping the readings older than the longest
// window. It returns the reading it recorded.
func (t *Tracker) Poll() (Reading, error) {
	reading, err := t.Read()
	if err != nil {
		return Reading{}, err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.readings = append(t.readings, reading)
	cutoff := reading.At.Add(-t.keep)
	first := 0
	for first < len(t.readings) && t.readings[first].At.Before(cutoff) {
		first++
	}
	if first > 0 {
		t.readings = append([]Reading(nil), t.readings[first:]...)
	}
	return reading, nil
}

// Snapshot polls once and returns the counters now with the rates over each window.
func (t *Tracker) Snapshot() Snapshot {
	now, err := t.Poll()
	if err != nil {
		return Snapshot{At: time.Now(), Phase: "unknown", Err: err.Error()}
	}
	snapshot := Snapshot{
		At:            now.At,
		Phase:         now.Phase,
		Streams:       now.Streams,
		StreamsEnc:    now.StreamsEnc,
		GateInFlight:  now.GateInFlight,
		GateWaiting:   now.GateWaiting,
		EncodeTokens:  now.EncodeTokens,
		DecodeTokens:  now.DecodeTokens,
		ContextTokens: now.ContextTokens,
		MaxContext:    now.MaxContext,
		EngineUp:      now.EngineUp,
	}
	if now.MaxContext > 0 {
		snapshot.ContextPct = 100 * float64(now.ContextTokens) / float64(now.MaxContext)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, window := range Windows {
		snapshot.Rates = append(snapshot.Rates, t.rateLocked(window))
	}
	return snapshot
}

// rateLocked computes the rates over one window from the oldest reading that is at least as old as the
// window, and the newest reading. The caller holds the lock.
func (t *Tracker) rateLocked(window time.Duration) Rate {
	rate := Rate{WindowS: window.Seconds()}
	if len(t.readings) < 2 {
		return rate
	}
	newest := t.readings[len(t.readings)-1]
	// the oldest reading that is at least window old; if none is that old, the window is not covered
	var oldest *Reading
	for i := len(t.readings) - 1; i >= 0; i-- {
		if newest.At.Sub(t.readings[i].At) >= window {
			oldest = &t.readings[i]
			break
		}
	}
	if oldest == nil {
		return rate
	}
	elapsed := newest.At.Sub(oldest.At).Seconds()
	if elapsed <= 0 {
		return rate
	}
	rate.Samples = len(t.readings)
	rate.EncodeTPS = float64(newest.EncodeTokens-oldest.EncodeTokens) / elapsed
	rate.DecodeTPS = float64(newest.DecodeTokens-oldest.DecodeTokens) / elapsed
	return rate
}

// Line returns one line per window, for the CLI: the phase, the streams, and the 2 rates.
func (snapshot Snapshot) Line() string {
	if snapshot.Err != "" {
		return "metrics: " + snapshot.Err
	}
	text := fmt.Sprintf("phase %s, streams %d (encoding %d), context %d of %d tokens (%.1f%%)",
		snapshot.Phase, snapshot.Streams, snapshot.StreamsEnc,
		snapshot.ContextTokens, snapshot.MaxContext, snapshot.ContextPct)
	for _, rate := range snapshot.Rates {
		if rate.Samples == 0 {
			text += fmt.Sprintf("\n  %4.1f s: not covered yet", rate.WindowS)
			continue
		}
		text += fmt.Sprintf("\n  %4.1f s: encode %8.1f tok/s, decode %8.1f tok/s", rate.WindowS, rate.EncodeTPS, rate.DecodeTPS)
	}
	return text
}
