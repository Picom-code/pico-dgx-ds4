// Command dsgb10-runner-sweep-go measures how the line's throughput changes as the number of concurrent
// streams rises: it sends 1, 2, 4, ... up to 32 chat completions at the same time, reads the tokens per
// second the line served over each step from the front-end's own counters, and repeats the whole sweep
// with a longer prompt so that the operator can see whether a longer context moves the point where
// throughput stops rising.
package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/sweep"
)

func main() {
	settings := sweep.Settings{
		BaseURL:      envOr("DSGB10_BASE_URL", "http://127.0.0.1:18893"),
		APIKey:       envOr("DSGB10_API_KEY", ""),
		Model:        envOr("DSGB10_MODEL", "dsgb10"),
		Levels:       sweep.DefaultLevels(),
		PromptTokens: []int{512, 8192},
		MaxTokens:    envInt("DSGB10_SWEEP_MAX_TOKENS", 128),
		SettleS:      envFloat("DSGB10_SWEEP_SETTLE_S", 1.0),
		Timeout:      time.Duration(envInt("DSGB10_SWEEP_TIMEOUT_S", 600)) * time.Second,
	}
	asJSON := false
	for _, arg := range os.Args[1:] {
		switch {
		case arg == "--json":
			asJSON = true
		case arg == "--help" || arg == "-h":
			usage()
			return
		case strings.HasPrefix(arg, "--base-url="):
			settings.BaseURL = strings.TrimPrefix(arg, "--base-url=")
		case strings.HasPrefix(arg, "--api-key="):
			settings.APIKey = strings.TrimPrefix(arg, "--api-key=")
		case strings.HasPrefix(arg, "--model="):
			settings.Model = strings.TrimPrefix(arg, "--model=")
		case strings.HasPrefix(arg, "--levels="):
			levels, err := parseLevels(strings.TrimPrefix(arg, "--levels="))
			if err != nil {
				fmt.Fprintf(os.Stderr, "dsgb10-runner-sweep-go: %v\n", err)
				os.Exit(2)
			}
			settings.Levels = levels
		case strings.HasPrefix(arg, "--prompt-tokens="):
			prompts, err := parseLevels(strings.TrimPrefix(arg, "--prompt-tokens="))
			if err != nil {
				fmt.Fprintf(os.Stderr, "dsgb10-runner-sweep-go: %v\n", err)
				os.Exit(2)
			}
			settings.PromptTokens = prompts
		case strings.HasPrefix(arg, "--max-tokens="):
			value, err := strconv.Atoi(strings.TrimPrefix(arg, "--max-tokens="))
			if err != nil || value <= 0 {
				fmt.Fprintf(os.Stderr, "dsgb10-runner-sweep-go: --max-tokens takes a positive count\n")
				os.Exit(2)
			}
			settings.MaxTokens = value
		default:
			fmt.Fprintf(os.Stderr, "dsgb10-runner-sweep-go: unknown argument %s\n", arg)
			os.Exit(2)
		}
	}
	report := sweep.Sweep(settings)
	if asJSON {
		text, err := report.JSON()
		if err != nil {
			fmt.Fprintf(os.Stderr, "dsgb10-runner-sweep-go: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(string(text))
	} else {
		fmt.Print(report.Table())
	}
	if report.Err != "" {
		os.Exit(1)
	}
	for _, run := range report.Runs {
		for _, step := range run.Steps {
			if step.Failed > 0 {
				os.Exit(1)
			}
		}
	}
}

// parseLevels reads a comma-separated list of positive counts, such as "1,2,4,8".
func parseLevels(text string) ([]int, error) {
	fields := strings.Split(text, ",")
	values := make([]int, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		value, err := strconv.Atoi(field)
		if err != nil || value <= 0 {
			return nil, fmt.Errorf("%q is not a positive count", field)
		}
		values = append(values, value)
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("no count given")
	}
	return values, nil
}

// envOr returns the environment's value of key, or fallback when it is unset or empty.
func envOr(key string, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}

// envInt returns the environment's value of key as a count, or fallback when it is unset or not a count.
func envInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

// envFloat returns the environment's value of key as a number, or fallback when it is unset or not one.
func envFloat(key string, fallback float64) float64 {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

// usage prints what the command does.
func usage() {
	fmt.Print(`dsgb10-runner-sweep-go: how throughput changes as concurrent streams rise

  dsgb10-runner-sweep-go                     the sweep as a table
  dsgb10-runner-sweep-go --json              the sweep as one JSON object

  --base-url=URL          the front-end (default http://127.0.0.1:18893)
  --api-key=KEY           the API key, when the front-end asks for one
  --model=NAME            the model name to ask for (default dsgb10)
  --levels=1,2,4,8,16,32  the concurrency levels, in rising order
  --prompt-tokens=512,8192  the prompt lengths, one sweep each
  --max-tokens=128        the answer length each stream asks for

Settings, from the environment: DSGB10_BASE_URL, DSGB10_API_KEY, DSGB10_MODEL,
DSGB10_SWEEP_MAX_TOKENS, DSGB10_SWEEP_SETTLE_S, DSGB10_SWEEP_TIMEOUT_S.

Each step sends its streams at the same time and reports the tokens per second the line served over
that step, read from the front-end's /metrics counters before and after the step. Exit 1 when a stream
failed or the counters could not be read.
`)
}
