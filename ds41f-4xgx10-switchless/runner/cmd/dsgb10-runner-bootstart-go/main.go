// Command dsgb10-runner-bootstart-go starts the line at boot: it waits for the ring's boxes to answer
// SSH, then starts the supervisor. systemd runs it as the unit's ExecStartPre, and an operator can run
// it by hand after a reboot.
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/bootstart"
	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/fleetconf"
)

func main() {
	for _, arg := range os.Args[1:] {
		switch arg {
		case "--help", "-h":
			usage()
			return
		default:
			fmt.Fprintf(os.Stderr, "dsgb10-runner-bootstart-go: unknown argument %s\n", arg)
			os.Exit(2)
		}
	}
	settings := bootstart.DefaultSettings()
	fleet, err := fleetconf.ReadFleet(settings.FleetPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-bootstart-go: %v\n", err)
		os.Exit(2)
	}
	workers, err := fleet.Workers()
	if err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-bootstart-go: %v\n", err)
		os.Exit(2)
	}
	serviceDir, err := fleet.ServiceDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-bootstart-go: %v\n", err)
		os.Exit(2)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-bootstart-go: %v\n", err)
		os.Exit(2)
	}
	settings.Workers = workers
	settings.Home = fleetconf.Local(serviceDir, 0, home)
	settings.ServerBin = envOr("DSGB10_SERVER_BIN", filepath.Join(home, "dsgb10", "bin", "dsgb10-runner-server-go"))
	logf := func(format string, args ...any) {
		fmt.Printf(format+"\n", args...)
	}
	if missing := bootstart.WaitForWorkers(settings, logf); len(missing) > 0 {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-bootstart-go: %v did not answer SSH within %d s; the engine starts anyway, and the supervisor restarts it until the boxes answer\n", missing, settings.WaitS)
	}
	if _, err := bootstart.Start(settings, logf); err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-bootstart-go: %v\n", err)
		os.Exit(1)
	}
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
	fmt.Print(`dsgb10-runner-bootstart-go: start the line at boot

  dsgb10-runner-bootstart-go    wait for the ring's boxes to answer SSH, then start the supervisor

Settings, from the environment:
  FLEET (fleet/homelab.conf)     the fleet file: the boxes, their addresses and directories
  DSGB10_SERVER_BIN              the supervisor binary (~/dsgb10/bin/dsgb10-runner-server-go)
  BOOT_WAIT_S (300)              how long to wait for the boxes to answer SSH
  BOOT_POLL_S (5)                how often to ask a box that has not answered
`)
}
