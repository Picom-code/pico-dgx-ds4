// Package bootstart starts the line at boot: it waits for the boxes of the ring to answer SSH, then
// starts the supervisor, and it reports what it did. It is the Go port of the boot starter that the
// dgx5-8 supervisor unit ran, and systemd runs it as the unit's ExecStartPre.
package bootstart

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Settings is what a boot start reads: the boxes to wait for, and how long to wait for them.
type Settings struct {
	Workers   []string
	WaitS     int64
	PollS     int64
	Home      string
	ServerBin string
	FleetPath string
	LogPath   string
}

// DefaultSettings returns the boot starter's settings, with the environment's overrides.
func DefaultSettings() Settings {
	return Settings{
		WaitS:     envInt64("BOOT_WAIT_S", 300),
		PollS:     envInt64("BOOT_POLL_S", 5),
		FleetPath: envOr("FLEET", "fleet/homelab.conf"),
	}
}

// WaitForWorkers waits until every worker answers SSH, and returns the boxes that did not answer within
// WaitS. A box that answers is reported once, so the log shows how long each took.
func WaitForWorkers(settings Settings, logf func(string, ...any)) []string {
	pending := make(map[string]bool, len(settings.Workers))
	for _, worker := range settings.Workers {
		pending[worker] = true
	}
	deadline := time.Now().Add(time.Duration(settings.WaitS) * time.Second)
	for len(pending) > 0 && time.Now().Before(deadline) {
		for _, worker := range settings.Workers {
			if !pending[worker] {
				continue
			}
			cmd := exec.Command("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", worker, "true")
			if err := cmd.Run(); err == nil {
				delete(pending, worker)
				logf("boot: %s answers SSH", worker)
			}
		}
		if len(pending) == 0 {
			break
		}
		time.Sleep(time.Duration(settings.PollS) * time.Second)
	}
	missing := make([]string, 0, len(pending))
	for _, worker := range settings.Workers {
		if pending[worker] {
			missing = append(missing, worker)
		}
	}
	return missing
}

// Start starts the supervisor as a detached process, and returns its pid. The supervisor writes its own
// pid file, so a start that finds one already running is not started twice.
func Start(settings Settings, logf func(string, ...any)) (int, error) {
	pidPath := filepath.Join(settings.Home, "run", "supervisor.pid")
	if text, err := os.ReadFile(pidPath); err == nil {
		pid, err := strconv.Atoi(strings.TrimSpace(string(text)))
		if err == nil && pid > 0 && processRuns(pid) {
			logf("boot: the supervisor already runs (pid %d)", pid)
			return pid, nil
		}
	}
	if err := os.MkdirAll(filepath.Join(settings.Home, "run"), 0o700); err != nil {
		return 0, fmt.Errorf("cannot make %s: %w", filepath.Join(settings.Home, "run"), err)
	}
	if err := os.MkdirAll(filepath.Join(settings.Home, "logs"), 0o755); err != nil {
		return 0, fmt.Errorf("cannot make %s: %w", filepath.Join(settings.Home, "logs"), err)
	}
	logPath := filepath.Join(settings.Home, "logs", "boot.log")
	logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, fmt.Errorf("cannot open %s: %w", logPath, err)
	}
	defer logFile.Close()
	// The supervisor runs in the fleet file's directory, so that a relative FLEET still names the file.
	// Every path it reads is made absolute here first, because a relative path would resolve against that
	// directory instead of the one the boot starter was run from.
	fleetPath, err := filepath.Abs(settings.FleetPath)
	if err != nil {
		return 0, fmt.Errorf("cannot resolve %s: %w", settings.FleetPath, err)
	}
	serverBin, err := filepath.Abs(settings.ServerBin)
	if err != nil {
		return 0, fmt.Errorf("cannot resolve %s: %w", settings.ServerBin, err)
	}
	cmd := exec.Command(serverBin)
	cmd.Dir = filepath.Dir(fleetPath)
	cmd.Env = append(os.Environ(), "FLEET="+fleetPath)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = detachedAttr()
	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("cannot start %s: %w", serverBin, err)
	}
	pid := cmd.Process.Pid
	logf("boot: the supervisor started (pid %d), log %s", pid, logPath)
	return pid, nil
}

// processRuns reports whether the process runs.
func processRuns(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return process.Signal(syscallSignalZero()) == nil
}

// envOr returns the environment's value of key, or fallback when it is unset or empty.
func envOr(key string, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}

// envInt64 returns the environment's value of key as an integer, or fallback when it is unset or not a
// number.
func envInt64(key string, fallback int64) int64 {
	text := os.Getenv(key)
	if text == "" {
		return fallback
	}
	value, err := strconv.ParseInt(text, 10, 64)
	if err != nil {
		return fallback
	}
	return value
}
