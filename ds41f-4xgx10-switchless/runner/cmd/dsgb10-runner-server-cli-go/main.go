// Command dsgb10-runner-server-cli-go reports the line's state in the terminal: whether it serves, how
// long it has run, how often it restarted, the front-end's /health code, and the last 20 lines of the
// supervisor's log. It reads the files that the supervisor writes, so it needs no socket.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/fleetconf"
	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/linestate"
)

func main() {
	asJSON := false
	for _, arg := range os.Args[1:] {
		switch arg {
		case "--json":
			asJSON = true
		case "--help", "-h":
			usage()
			return
		default:
			fmt.Fprintf(os.Stderr, "dsgb10-runner-server-cli-go: unknown argument %s\n", arg)
			os.Exit(2)
		}
	}
	paths, err := readPaths()
	if err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-server-cli-go: %v\n", err)
		os.Exit(2)
	}
	state := paths.Read()
	if asJSON {
		text, err := state.JSON()
		if err != nil {
			fmt.Fprintf(os.Stderr, "dsgb10-runner-server-cli-go: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(string(text))
		return
	}
	printState(state)
	if !state.HealthOK {
		os.Exit(1)
	}
}

// readPaths reads the fleet file and returns the files of the line's service directory.
func readPaths() (linestate.Paths, error) {
	fleetPath := os.Getenv("FLEET")
	if fleetPath == "" {
		fleetPath = "fleet/homelab.conf"
	}
	fleet, err := fleetconf.ReadFleet(fleetPath)
	if err != nil {
		return linestate.Paths{}, err
	}
	serviceDir, err := fleet.ServiceDir()
	if err != nil {
		return linestate.Paths{}, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return linestate.Paths{}, err
	}
	addrs := splitAddrs(envOr("DSGB10_ADDR", "127.0.0.1"))
	return linestate.Paths{
		Home:  fleetconf.Local(serviceDir, 0, home),
		Addrs: addrs,
		Port:  envOr("DSGB10_PORT", "18893"),
	}, nil
}

// printState writes the line's state as one block of lines: whether it serves, how long it has run, how
// often it restarted, the front-end's /health code, and the last 20 lines of the supervisor's log.
func printState(state linestate.State) {
	fmt.Printf("service: %s since %s: %s\n", state.State, state.Since.Format("2006-01-02 15:04:05"), state.What)
	fmt.Printf("  uptime: %s\n", formatUptime(state.UptimeS))
	if state.Restarts != "" {
		fmt.Printf("  %s\n", state.Restarts)
	}
	if state.RingOwner != "" {
		fmt.Printf("  ring: %s\n", state.RingOwner)
	}
	codes := make([]string, 0, len(state.HealthCodes))
	for _, code := range state.HealthCodes {
		codes = append(codes, fmt.Sprintf("%d", code))
	}
	verdict := "not serving"
	if state.HealthOK {
		verdict = "the engine is connected"
	}
	fmt.Printf("  front-end: %s, pid %d, /health %s (%s)\n", strings.Join(state.URLs, " "), state.FrontPid, strings.Join(codes, " "), verdict)
	if state.Rank0Pid > 0 {
		fmt.Printf("  rank 0: pid %d, up %ds, MemAvailable %d GiB\n", state.Rank0Pid, state.Rank0UptimeS, state.MemAvailGiB)
	} else {
		fmt.Printf("  rank 0: not running, MemAvailable %d GiB\n", state.MemAvailGiB)
	}
	fmt.Printf("  last %d lines of service.log:\n", len(state.LogTail))
	for _, line := range state.LogTail {
		fmt.Printf("    %s\n", line)
	}
}

// formatUptime returns a duration as days, hours, minutes and seconds, so that a long run reads at once.
func formatUptime(seconds int64) string {
	if seconds < 0 {
		seconds = 0
	}
	days := seconds / 86400
	hours := (seconds % 86400) / 3600
	minutes := (seconds % 3600) / 60
	rest := seconds % 60
	if days > 0 {
		return fmt.Sprintf("%dd %dh %dm %ds", days, hours, minutes, rest)
	}
	if hours > 0 {
		return fmt.Sprintf("%dh %dm %ds", hours, minutes, rest)
	}
	return fmt.Sprintf("%dm %ds", minutes, rest)
}

// splitAddrs splits an address list on spaces and commas.
func splitAddrs(text string) []string {
	fields := strings.FieldsFunc(text, func(r rune) bool { return r == ' ' || r == ',' })
	addrs := make([]string, 0, len(fields))
	for _, field := range fields {
		if field != "" {
			addrs = append(addrs, field)
		}
	}
	return addrs
}

// envOr returns the environment's value of key, or fallback when it is unset or empty.
func envOr(key string, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}

// usage prints what the command does.
func usage() {
	fmt.Print(`dsgb10-runner-server-cli-go: the line's state in the terminal

  dsgb10-runner-server-cli-go           the state as lines; exit 1 when the line does not serve
  dsgb10-runner-server-cli-go --json    the state as one JSON object

Settings, from the environment: FLEET (fleet/homelab.conf), DSGB10_ADDR, DSGB10_PORT.
`)
}

// unused keeps the json import when the flag is compiled out.
var _ = json.Marshal
