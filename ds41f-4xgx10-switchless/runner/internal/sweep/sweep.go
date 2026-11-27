// Package sweep measures how the line's throughput changes as the number of concurrent streams rises.
//
// A sweep sends 1, 2, 4, ... up to a maximum number of chat completions at the same time, waits for all
// of them to finish, and reports the tokens per second the line served over that step. The rate is
// computed from the front-end's own counters, read before the step and after it, so it counts the tokens
// the engine produced rather than the tokens a client happened to receive.
//
// The same sweep is run twice: once with a short prompt and once with a long one. The 2 runs answer the
// operator's question of where throughput stops rising, and whether a longer context moves that point.
package sweep

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// Step is one concurrency level of a sweep: the streams sent together and what the line served.
type Step struct {
	Streams int `json:"streams"`
	// OK is how many streams answered 200; Failed is how many did not.
	OK     int `json:"ok"`
	Failed int `json:"failed"`
	// WallS is how long the step took, from the first submit to the last answer.
	WallS float64 `json:"wall_s"`
	// EncodeTokens and DecodeTokens are the tokens the line produced during the step, from its counters.
	EncodeTokens int64 `json:"encode_tokens"`
	DecodeTokens int64 `json:"decode_tokens"`
	// EncodeTPS and DecodeTPS are those counts divided by WallS.
	EncodeTPS float64 `json:"encode_tps"`
	DecodeTPS float64 `json:"decode_tps"`
	// TotalTPS is the tokens of both kinds per second: the line's whole output rate.
	TotalTPS float64 `json:"total_tps"`
	// PerStreamTPS is DecodeTPS divided by Streams: what one stream saw.
	PerStreamTPS float64 `json:"per_stream_tps"`
	// PeakContext is the largest context the line held during the step, in tokens.
	PeakContext int64 `json:"peak_context"`
	// Error is the first failure's message, when a stream failed.
	Error string `json:"error,omitempty"`
}

// Run is one sweep at one prompt length.
type Run struct {
	// PromptTokens is the prompt length the run asked for, in tokens.
	PromptTokens int `json:"prompt_tokens"`
	// MaxTokens is the answer length each stream asked for.
	MaxTokens int `json:"max_tokens"`
	// Steps is one entry per concurrency level, in rising order.
	Steps []Step `json:"steps"`
	// Best is the concurrency level with the highest TotalTPS.
	Best int `json:"best_streams"`
	// BestTPS is that level's TotalTPS.
	BestTPS float64 `json:"best_tps"`
}

// Report is what a sweep writes: the 2 runs and the settings they ran with.
type Report struct {
	At         time.Time `json:"at"`
	BaseURL    string    `json:"base_url"`
	Model      string    `json:"model"`
	MaxContext int64     `json:"max_context"`
	Runs       []Run     `json:"runs"`
	Err        string    `json:"error,omitempty"`
}

// Settings are the knobs of one sweep.
type Settings struct {
	BaseURL string
	APIKey  string
	Model   string
	// Levels are the concurrency levels to run, in rising order.
	Levels []int
	// PromptTokens is the prompt length of the first run and of the second run.
	PromptTokens []int
	// MaxTokens is the answer length each stream asks for.
	MaxTokens int
	// SettleS is how long the sweep waits after a step before reading the counters, so that the last
	// tokens of the step are counted.
	SettleS float64
	// Timeout is how long one stream may take.
	Timeout time.Duration
}

// DefaultLevels returns the concurrency levels the operator asked for: 1 to 32, doubling.
func DefaultLevels() []int {
	return []int{1, 2, 4, 8, 16, 32}
}

// counters is the part of the front-end's /metrics answer that a sweep reads.
type counters struct {
	EncodeTokens  int64 `json:"encode_tokens"`
	DecodeTokens  int64 `json:"decode_tokens"`
	ContextTokens int64 `json:"context_tokens"`
	MaxContext    int64 `json:"max_context"`
}

// readCounters reads the front-end's counters once.
func readCounters(client *http.Client, settings Settings) (counters, error) {
	request, err := http.NewRequest(http.MethodGet, settings.BaseURL+"/metrics", nil)
	if err != nil {
		return counters{}, err
	}
	if settings.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+settings.APIKey)
	}
	response, err := client.Do(request)
	if err != nil {
		return counters{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return counters{}, fmt.Errorf("/metrics answered %d", response.StatusCode)
	}
	var got counters
	if err := json.NewDecoder(response.Body).Decode(&got); err != nil {
		return counters{}, fmt.Errorf("/metrics is not JSON: %w", err)
	}
	return got, nil
}

// prompt returns a prompt of about the requested length in tokens. English text runs about 4 characters
// to a token, and the sweep says the length it asked for rather than the length it got.
func prompt(tokens int) string {
	const word = "the line serves tokens "
	var builder strings.Builder
	builder.WriteString("Reply with one short sentence.\n\n")
	for builder.Len() < tokens*4 {
		builder.WriteString(word)
	}
	return builder.String()
}

// stream sends one chat completion and returns the answer's token count, or an error.
func stream(client *http.Client, settings Settings, promptText string) (int, error) {
	body := map[string]any{
		"model":      settings.Model,
		"messages":   []map[string]string{{"role": "user", "content": promptText}},
		"max_tokens": settings.MaxTokens,
		"stream":     false,
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return 0, err
	}
	request, err := http.NewRequest(http.MethodPost, settings.BaseURL+"/v1/chat/completions", bytes.NewReader(encoded))
	if err != nil {
		return 0, err
	}
	request.Header.Set("Content-Type", "application/json")
	if settings.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+settings.APIKey)
	}
	response, err := client.Do(request)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	text, err := io.ReadAll(response.Body)
	if err != nil {
		return 0, err
	}
	if response.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("%d: %s", response.StatusCode, firstLine(string(text)))
	}
	var answer struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(text, &answer); err != nil {
		return 0, fmt.Errorf("the answer is not JSON: %w", err)
	}
	return answer.Usage.CompletionTokens, nil
}

// firstLine returns the first line of text, shortened, for an error message.
func firstLine(text string) string {
	line, _, _ := strings.Cut(strings.TrimSpace(text), "\n")
	if len(line) > 200 {
		line = line[:200]
	}
	return line
}

// RunOne runs every level of a sweep at one prompt length.
func RunOne(settings Settings, promptTokens int) Run {
	client := &http.Client{Timeout: settings.Timeout}
	run := Run{PromptTokens: promptTokens, MaxTokens: settings.MaxTokens}
	promptText := prompt(promptTokens)
	for _, level := range settings.Levels {
		step := runStep(client, settings, promptText, level)
		run.Steps = append(run.Steps, step)
		if step.TotalTPS > run.BestTPS {
			run.BestTPS = step.TotalTPS
			run.Best = step.Streams
		}
	}
	return run
}

// runStep sends level streams at the same time and measures what the line served.
func runStep(client *http.Client, settings Settings, promptText string, level int) Step {
	step := Step{Streams: level}
	before, err := readCounters(client, settings)
	if err != nil {
		step.Error = "cannot read the counters before the step: " + err.Error()
		return step
	}
	// The peak context is sampled while the step runs, so that the step reports the largest context the
	// line held rather than the context left at the end.
	var peak int64
	var peakMu sync.Mutex
	stop := make(chan struct{})
	var sampler sync.WaitGroup
	sampler.Add(1)
	go func() {
		defer sampler.Done()
		for {
			select {
			case <-stop:
				return
			case <-time.After(100 * time.Millisecond):
			}
			got, err := readCounters(client, settings)
			if err != nil {
				continue
			}
			peakMu.Lock()
			if got.ContextTokens > peak {
				peak = got.ContextTokens
			}
			peakMu.Unlock()
		}
	}()

	var wait sync.WaitGroup
	var mu sync.Mutex
	start := time.Now()
	for index := 0; index < level; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := stream(client, settings, promptText)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				step.Failed++
				if step.Error == "" {
					step.Error = err.Error()
				}
				return
			}
			step.OK++
		}()
	}
	wait.Wait()
	step.WallS = time.Since(start).Seconds()
	close(stop)
	sampler.Wait()

	// the counters are read after a settle, so that the tokens of the last answers are counted
	time.Sleep(time.Duration(settings.SettleS * float64(time.Second)))
	after, err := readCounters(client, settings)
	if err != nil {
		step.Error = "cannot read the counters after the step: " + err.Error()
		return step
	}
	step.EncodeTokens = after.EncodeTokens - before.EncodeTokens
	step.DecodeTokens = after.DecodeTokens - before.DecodeTokens
	if step.WallS > 0 {
		step.EncodeTPS = float64(step.EncodeTokens) / step.WallS
		step.DecodeTPS = float64(step.DecodeTokens) / step.WallS
		step.TotalTPS = float64(step.EncodeTokens+step.DecodeTokens) / step.WallS
		step.PerStreamTPS = step.DecodeTPS / float64(level)
	}
	peakMu.Lock()
	step.PeakContext = peak
	peakMu.Unlock()
	return step
}

// Sweep runs the sweep at every prompt length and returns the report.
func Sweep(settings Settings) Report {
	report := Report{At: time.Now(), BaseURL: settings.BaseURL, Model: settings.Model}
	client := &http.Client{Timeout: settings.Timeout}
	if got, err := readCounters(client, settings); err == nil {
		report.MaxContext = got.MaxContext
	} else {
		report.Err = "cannot read /metrics: " + err.Error()
		return report
	}
	for _, promptTokens := range settings.PromptTokens {
		report.Runs = append(report.Runs, RunOne(settings, promptTokens))
	}
	return report
}

// Table returns the report as lines of text, one line per step, for the terminal.
func (report Report) Table() string {
	if report.Err != "" {
		return "sweep: " + report.Err
	}
	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("sweep of %s at %s, model %s, context %d tokens\n",
		report.BaseURL, report.At.Format("2006-01-02 15:04:05"), report.Model, report.MaxContext))
	for _, run := range report.Runs {
		builder.WriteString(fmt.Sprintf("\nprompt %d tokens, answer up to %d tokens\n", run.PromptTokens, run.MaxTokens))
		builder.WriteString("  streams   ok  fail   wall s   encode tok/s   decode tok/s   total tok/s   per stream   peak ctx\n")
		for _, step := range run.Steps {
			builder.WriteString(fmt.Sprintf("  %7d %4d %5d %8.2f %14.1f %14.1f %13.1f %12.1f %10d\n",
				step.Streams, step.OK, step.Failed, step.WallS,
				step.EncodeTPS, step.DecodeTPS, step.TotalTPS, step.PerStreamTPS, step.PeakContext))
			if step.Error != "" {
				builder.WriteString("           first failure: " + step.Error + "\n")
			}
		}
		builder.WriteString(fmt.Sprintf("  best: %d streams at %.1f tokens/s\n", run.Best, run.BestTPS))
	}
	return builder.String()
}

// JSON returns the report as one JSON object.
func (report Report) JSON() ([]byte, error) {
	return json.MarshalIndent(report, "", " ")
}

// sortLevels sorts and de-duplicates the levels, so that a sweep always rises.
func sortLevels(levels []int) []int {
	seen := make(map[int]bool, len(levels))
	kept := make([]int, 0, len(levels))
	for _, level := range levels {
		if level > 0 && !seen[level] {
			seen[level] = true
			kept = append(kept, level)
		}
	}
	sort.Ints(kept)
	return kept
}
