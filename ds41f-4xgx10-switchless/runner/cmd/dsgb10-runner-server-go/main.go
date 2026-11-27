// Command dsgb10-runner-server-go runs the line: the front-end dsfront and the four engine ranks, under
// one supervisor that holds the ring lock while it serves, restarts what fails with bounded backoff,
// and lends the ring to a job that waits for it. It is the Go port of engine/server/service.sh.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/supervisor"
)

func main() {
	if len(os.Args) > 1 && (os.Args[1] == "--help" || os.Args[1] == "-h") {
		usage()
		return
	}
	settings := supervisor.DefaultSettings()
	if err := settings.LoadFleet(); err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-server-go: %v\n", err)
		os.Exit(2)
	}
	if err := checkSettings(settings); err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-server-go: %v\n", err)
		os.Exit(2)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-signals
		cancel()
	}()
	runner := supervisor.New(settings)
	if err := runner.Run(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-server-go: %v\n", err)
		os.Exit(1)
	}
}

// checkSettings rejects the settings that the line cannot run with, and names the one it rejects.
func checkSettings(settings supervisor.Settings) error {
	if len(settings.Addrs) == 0 {
		return fmt.Errorf("DSGB10_ADDR is empty")
	}
	seen := make(map[string]bool, len(settings.Addrs))
	for _, addr := range settings.Addrs {
		switch addr {
		case "0.0.0.0", "::", "[::]", "*":
			return fmt.Errorf("will not listen on '%s': specific addresses only, never 0.0.0.0 (HTTP stays off public interfaces)", addr)
		}
		if seen[addr] {
			return fmt.Errorf("DSGB10_ADDR names '%s' twice", addr)
		}
		seen[addr] = true
	}
	if settings.Port == "8888" && os.Getenv("DSGB10_PROD") != "1" {
		return fmt.Errorf("port 8888 is the production endpoint: DSGB10_PROD=1")
	}
	info, err := os.Stat(settings.KeyFile)
	if err != nil {
		return fmt.Errorf("no API key file %s", settings.KeyFile)
	}
	if info.Mode().Perm()&0o077 != 0 {
		return fmt.Errorf("%s can be read by others: chmod 600 it", settings.KeyFile)
	}
	if _, err := os.Stat(settings.FrontBin); err != nil {
		return fmt.Errorf("no front-end binary %s (cargo build --release in frontend/)", settings.FrontBin)
	}
	if _, err := os.Stat(settings.EngineBin); err != nil {
		return fmt.Errorf("no engine binary %s (cmake --build engine/build)", settings.EngineBin)
	}
	return nil
}

// usage prints what the command does and the settings it reads.
func usage() {
	fmt.Print(`dsgb10-runner-server-go: the line under a supervisor (the Go port of engine/server/service.sh)

  dsgb10-runner-server-go            run until SIGTERM: the front-end, then the four ranks

Settings, from the environment:
  DSGB10_ADDR (127.0.0.1)        the front-end's addresses, separated by spaces or commas
  DSGB10_PORT (18893)            the front-end's port
  DSGB10_KEY_FILE                the API key file (mode 600)
  DSGB10_ENGINE_ARGS             the engine's arguments after --mode serve --socket <run/engine.sock>
  DSGB10_FRONT_ARGS              more of dsfront's arguments
  DSGB10_BIN, DSGB10_FRONT_BIN   the engine and the front-end binaries, frozen into <home>/bin at start
  DSGB10_YIELD_IDLE_S (600)      the ring is lent once no request has come for this long
  DSGB10_HOLD_S (600)            dsfront holds a request this long while the engine is down
  MIN_AVAIL_GIB (100)            a box with less MemAvailable fails the engine's start
  DSGB10_KEEP_GIB (4)            the kept sessions written at an engine stop
  READY_S (600)                  rank 0 must be listening this long after its start
  MAX_FAILS (5), FAIL_WINDOW (3600)  more failures than this within the window: give up
  FLEET (fleet/homelab.conf)     the fleet file: the boxes, their addresses and directories

Files: <home>/{bin,logs,run}; logs/service.log (this supervisor), rank.0.log and front.log.
`)
}
