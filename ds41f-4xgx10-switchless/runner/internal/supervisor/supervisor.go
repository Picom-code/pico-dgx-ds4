// Package supervisor runs the line: the front-end dsfront and the four engine ranks, under one process
// that holds the ring lock while it serves, restarts what fails with bounded backoff, and lends the
// ring to a job that waits for it. It is the Go port of engine/server/service.sh, and it keeps the same
// files in the same places, so that a line started by either one is read by the other.
package supervisor

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/fleetconf"
	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/ringlock"
)

// Settings is what a start reads: the addresses, the binaries, and the bounds of the supervisor.
type Settings struct {
	Addrs        []string
	Port         string
	KeyFile      string
	EngineArgs   []string
	FrontArgs    []string
	Note         string
	EngineBin    string
	FrontBin     string
	MinAvailGiB  int64
	YieldIdleS   int64
	HoldS        int64
	KeepGiB      string
	ReadyS       int64
	MaxFails     int
	FailWindowS  int64
	EngineDelays []int64
	FrontDelays  []int64
	Keep         int
	LogMiB       int64
	FleetPath    string
	Home         string
	Workers      []string
	Rank0Name    string
	RingLockPath string
}

// DefaultSettings returns the settings of engine/server/service.sh, with the environment's overrides.
func DefaultSettings() Settings {
	settings := Settings{
		Addrs:        splitAddrs(envOr("DSGB10_ADDR", "127.0.0.1")),
		Port:         envOr("DSGB10_PORT", "18893"),
		KeyFile:      envOr("DSGB10_KEY_FILE", filepath.Join(homeDir(), "dsgb10-data", "dev-api-key")),
		EngineArgs:   strings.Fields(envOr("DSGB10_ENGINE_ARGS", "--max-ctx 262144")),
		FrontArgs:    strings.Fields(os.Getenv("DSGB10_FRONT_ARGS")),
		Note:         os.Getenv("DSGB10_NOTE"),
		EngineBin:    os.Getenv("DSGB10_BIN"),
		FrontBin:     os.Getenv("DSGB10_FRONT_BIN"),
		MinAvailGiB:  envInt64("MIN_AVAIL_GIB", 100),
		YieldIdleS:   envInt64("DSGB10_YIELD_IDLE_S", 600),
		HoldS:        envInt64("DSGB10_HOLD_S", 600),
		KeepGiB:      envOr("DSGB10_KEEP_GIB", "4"),
		ReadyS:       envInt64("READY_S", 600),
		MaxFails:     int(envInt64("MAX_FAILS", 5)),
		FailWindowS:  envInt64("FAIL_WINDOW", 3600),
		EngineDelays: []int64{10, 30, 60, 120, 300},
		FrontDelays:  []int64{2, 5, 10, 30, 60},
		Keep:         int(envInt64("KEEP", 5)),
		LogMiB:       envInt64("LOG_MIB", 64),
		FleetPath:    envOr("FLEET", "fleet/homelab.conf"),
	}
	return settings
}

// LoadFleet reads the fleet file and fills the settings that come from it.
func (settings *Settings) LoadFleet() error {
	fleet, err := fleetconf.ReadFleet(settings.FleetPath)
	if err != nil {
		return err
	}
	serviceDir, err := fleet.ServiceDir()
	if err != nil {
		return err
	}
	ringLockPath, err := fleet.RingLock()
	if err != nil {
		return err
	}
	workers, err := fleet.Workers()
	if err != nil {
		return err
	}
	settings.Home = fleetconf.Local(serviceDir, 0, homeDir())
	settings.RingLockPath = fleetconf.Local(ringLockPath, 0, homeDir())
	settings.Workers = workers
	settings.Rank0Name = fleet.Rank0Name()
	return nil
}

// RunDir returns the service's run directory.
func (settings Settings) RunDir() string { return filepath.Join(settings.Home, "run") }

// LogDir returns the service's log directory.
func (settings Settings) LogDir() string { return filepath.Join(settings.Home, "logs") }

// BinDir returns the directory that holds the frozen binaries.
func (settings Settings) BinDir() string { return filepath.Join(settings.Home, "bin") }

// Supervisor runs the line until its context ends.
type Supervisor struct {
	Settings Settings
	Lock     *ringlock.Lock

	mutex      sync.Mutex
	engineCmd  *exec.Cmd
	frontCmd   *exec.Cmd
	engineAt   time.Time
	frontAt    time.Time
	ready      bool
	readyAt    time.Time
	lent       bool
	lentAt     time.Time
	reclaiming bool
	first      bool
	fails      []time.Time
	restarts   int
	stop       bool
}

// New returns a supervisor for the settings, with the ring lock named by the fleet file.
func New(settings Settings) *Supervisor {
	return &Supervisor{Settings: settings, Lock: ringlock.NewLock(settings.RingLockPath)}
}

// Log writes one line to the supervisor's log and to stdout.
func (supervisor *Supervisor) Log(format string, args ...any) {
	line := fmt.Sprintf("%s %s\n", time.Now().Format("2006-01-02 15:04:05"), fmt.Sprintf(format, args...))
	os.Stdout.WriteString(line)
	logPath := filepath.Join(supervisor.Settings.LogDir(), "service.log")
	file, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer file.Close()
	file.WriteString(line)
}

// SetState writes run/state: "<state> <epoch> <what>".
func (supervisor *Supervisor) SetState(state string, what string) {
	path := filepath.Join(supervisor.Settings.RunDir(), "state")
	line := fmt.Sprintf("%s %d %s\n", state, time.Now().Unix(), what)
	temporary := path + ".new"
	if err := os.WriteFile(temporary, []byte(line), 0o644); err != nil {
		return
	}
	os.Rename(temporary, path)
	supervisor.Log("state: %s %s", state, what)
}

// Run starts the front-end and the engine, then supervises them until the context ends.
func (supervisor *Supervisor) Run(ctx context.Context) error {
	settings := supervisor.Settings
	if err := os.MkdirAll(settings.RunDir(), 0o700); err != nil {
		return fmt.Errorf("cannot make %s: %w", settings.RunDir(), err)
	}
	if err := os.MkdirAll(settings.LogDir(), 0o755); err != nil {
		return fmt.Errorf("cannot make %s: %w", settings.LogDir(), err)
	}
	pidPath := filepath.Join(settings.RunDir(), "supervisor.pid")
	os.WriteFile(pidPath, []byte(strconv.Itoa(os.Getpid())+"\n"), 0o644)
	defer os.Remove(pidPath)
	os.WriteFile(filepath.Join(settings.RunDir(), "restarts"), []byte("0 restarts\n"), 0o644)
	supervisor.Log("supervisor pid %d: front-end %s, engine %s", os.Getpid(), strings.Join(supervisor.URLs(), " "), strings.Join(settings.EngineArgs, " "))
	if err := supervisor.freeze(); err != nil {
		supervisor.SetState("failed", "copying the binaries failed")
		return err
	}
	supervisor.startFront()
	if taken, err := supervisor.Lock.Take(); err == nil && taken {
		supervisor.Lock.WriteOwner(fmt.Sprintf("service: dsgb10-runner-server-go, front-end %s", strings.Join(supervisor.URLs(), " ")))
	} else {
		supervisor.lent = true
		supervisor.lentAt = time.Now()
		supervisor.first = true
		supervisor.Log("the ring is busy (%s): the engine starts once no job waits", supervisor.Lock.ReadOwner())
		supervisor.SetState("waiting", fmt.Sprintf("for the ring; requests wait at the front-end, up to %d s each", settings.HoldS))
	}
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	lastWorkerCheck := time.Now()
	for {
		select {
		case <-ctx.Done():
			supervisor.shutdown()
			return nil
		case <-ticker.C:
		}
		now := time.Now()
		supervisor.checkFront(now)
		supervisor.checkEngine(now)
		supervisor.checkRing(now)
		if supervisor.engineCmd == nil && !supervisor.lent {
			supervisor.startEngine()
		}
		if supervisor.engineCmd != nil && !supervisor.ready {
			socket := filepath.Join(settings.RunDir(), "engine.sock")
			if info, err := os.Stat(socket); err == nil && info.Mode()&os.ModeSocket != 0 {
				supervisor.ready = true
				supervisor.readyAt = now
				supervisor.Log("engine ready after %d s", int64(now.Sub(supervisor.engineAt).Seconds()))
				supervisor.SetState("serving", fmt.Sprintf("%s; engine up since %s", strings.Join(supervisor.URLs(), " "), supervisor.engineAt.Format("2006-01-02 15:04:05")))
			} else if int64(now.Sub(supervisor.engineAt).Seconds()) > settings.ReadyS {
				supervisor.Log("rank 0 not listening %d s after its start: SIGTERM", settings.ReadyS)
				supervisor.engineCmd.Process.Signal(syscall.SIGTERM)
			}
		}
		if now.Sub(lastWorkerCheck) >= 60*time.Second {
			lastWorkerCheck = now
			supervisor.checkWorkers()
			supervisor.rotateLive()
		}
	}
}

// URLs returns the front-end's /v1 URLs, one per address.
func (supervisor *Supervisor) URLs() []string {
	urls := make([]string, 0, len(supervisor.Settings.Addrs))
	for _, addr := range supervisor.Settings.Addrs {
		urls = append(urls, "http://"+addr+":"+supervisor.Settings.Port+"/v1")
	}
	return urls
}

// freeze copies the binaries into the service's bin directory, so that a later build does not change a
// running line.
func (supervisor *Supervisor) freeze() error {
	settings := supervisor.Settings
	if err := os.MkdirAll(settings.BinDir(), 0o755); err != nil {
		return fmt.Errorf("cannot make %s: %w", settings.BinDir(), err)
	}
	pairs := [][2]string{
		{settings.EngineBin, filepath.Join(settings.BinDir(), "dsgb10_rank")},
		{settings.FleetPath, filepath.Join(settings.BinDir(), "fleet.conf")},
		{settings.FrontBin, filepath.Join(settings.BinDir(), "dsfront")},
	}
	for _, pair := range pairs {
		if pair[0] == "" {
			continue
		}
		if err := copyFile(pair[0], pair[1]); err != nil {
			return err
		}
	}
	return nil
}

// startEngine starts the four ranks: ranks 1 to 3 over SSH, then rank 0 here.
func (supervisor *Supervisor) startEngine() {
	settings := supervisor.Settings
	supervisor.SetState("starting", "the engine: page cache, memory check, four ranks")
	if available := memAvailableGiB(); available < settings.MinAvailGiB {
		supervisor.Log("not starting: MemAvailable %d GiB under %d GiB (another model loaded?)", available, settings.MinAvailGiB)
		supervisor.failure("the engine did not start")
		return
	}
	socket := filepath.Join(settings.RunDir(), "engine.sock")
	args := []string{"--mode", "serve", "--socket", socket, "--disk-tag", "service", "--keep-gib", settings.KeepGiB}
	args = append(args, settings.EngineArgs...)
	for index, worker := range settings.Workers {
		rank := index + 1
		command := fmt.Sprintf("cd %s && setsid nohup bin/dsgb10_rank --rank %d %s >> logs/rank.%d.log 2>&1 < /dev/null &",
			settings.Home, rank, strings.Join(args, " "), rank)
		cmd := exec.Command("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", worker, command)
		if err := cmd.Run(); err != nil {
			supervisor.Log("starting rank %d on %s failed: %v", rank, worker, err)
			supervisor.failure("the engine did not start")
			return
		}
	}
	supervisor.rotate(filepath.Join(settings.LogDir(), "rank.0.log"))
	os.Remove(socket)
	enginePath := filepath.Join(settings.BinDir(), "dsgb10_rank")
	cmd := exec.Command(enginePath, append([]string{"--rank", "0"}, args...)...)
	logFile, err := os.OpenFile(filepath.Join(settings.LogDir(), "rank.0.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		supervisor.Log("cannot open rank.0.log: %v", err)
		supervisor.failure("the engine did not start")
		return
	}
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		logFile.Close()
		supervisor.Log("starting rank 0 failed: %v", err)
		supervisor.failure("the engine did not start")
		return
	}
	supervisor.mutex.Lock()
	supervisor.engineCmd = cmd
	supervisor.engineAt = time.Now()
	supervisor.ready = false
	supervisor.mutex.Unlock()
	os.WriteFile(filepath.Join(settings.RunDir(), "rank0.pid"), []byte(strconv.Itoa(cmd.Process.Pid)+"\n"), 0o644)
	supervisor.Log("engine started: rank 0 pid %d, %s %s", cmd.Process.Pid, enginePath, strings.Join(args, " "))
}

// stopEngine stops rank 0 first: on SIGTERM it ends the request in flight, keeps the sessions, and
// stops ranks 1 to 3.
func (supervisor *Supervisor) stopEngine() {
	supervisor.mutex.Lock()
	cmd := supervisor.engineCmd
	supervisor.mutex.Unlock()
	if cmd != nil && cmd.Process != nil {
		cmd.Process.Signal(syscall.SIGTERM)
		// Wait reaps the child, so a child that exits leaves no zombie: a zombie still answers
		// signal 0, and polling signal 0 alone would wait the full 60 s for every stop.
		exited := make(chan struct{})
		go func() {
			cmd.Wait()
			close(exited)
		}()
		select {
		case <-exited:
			supervisor.Log("rank 0 exited")
		case <-time.After(60 * time.Second):
			supervisor.Log("rank 0 still running 60 s after SIGTERM: kill -9 (nothing kept)")
			cmd.Process.Kill()
			<-exited
			supervisor.Log("rank 0 exited")
		}
	}
	supervisor.mutex.Lock()
	supervisor.engineCmd = nil
	supervisor.ready = false
	supervisor.mutex.Unlock()
	os.Remove(filepath.Join(supervisor.Settings.RunDir(), "rank0.pid"))
	supervisor.reapAll(10)
}

// startFront starts dsfront, which holds requests while the engine is down.
func (supervisor *Supervisor) startFront() {
	settings := supervisor.Settings
	supervisor.rotate(filepath.Join(settings.LogDir(), "front.log"))
	args := make([]string, 0, 16)
	args = append(args, "serve")
	for _, addr := range settings.Addrs {
		args = append(args, "--listen", addr+":"+settings.Port)
	}
	args = append(args,
		"--engine-socket", filepath.Join(settings.RunDir(), "engine.sock"),
		"--api-key-file", settings.KeyFile,
		"--admin-socket", filepath.Join(settings.RunDir(), "front.admin"),
		"--hold-s", strconv.FormatInt(settings.HoldS, 10),
		"--table-file", filepath.Join(settings.RunDir(), "front.table"),
	)
	args = append(args, settings.FrontArgs...)
	frontPath := filepath.Join(settings.BinDir(), "dsfront")
	cmd := exec.Command(frontPath, args...)
	logFile, err := os.OpenFile(filepath.Join(settings.LogDir(), "front.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		supervisor.Log("cannot open front.log: %v", err)
		return
	}
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		logFile.Close()
		supervisor.Log("starting the front-end failed: %v", err)
		return
	}
	supervisor.mutex.Lock()
	supervisor.frontCmd = cmd
	supervisor.frontAt = time.Now()
	supervisor.mutex.Unlock()
	os.WriteFile(filepath.Join(settings.RunDir(), "front.pid"), []byte(strconv.Itoa(cmd.Process.Pid)+"\n"), 0o644)
	supervisor.Log("front-end started: pid %d, %s", cmd.Process.Pid, strings.Join(supervisor.URLs(), " "))
}

// stopFront stops dsfront.
func (supervisor *Supervisor) stopFront() {
	supervisor.mutex.Lock()
	cmd := supervisor.frontCmd
	supervisor.mutex.Unlock()
	if cmd != nil && cmd.Process != nil {
		cmd.Process.Signal(syscall.SIGTERM)
		exited := make(chan struct{})
		go func() {
			cmd.Wait()
			close(exited)
		}()
		select {
		case <-exited:
		case <-time.After(10 * time.Second):
			supervisor.Log("front-end still running 10 s after SIGTERM: kill -9")
			cmd.Process.Kill()
			<-exited
		}
	}
	supervisor.mutex.Lock()
	supervisor.frontCmd = nil
	supervisor.mutex.Unlock()
	os.Remove(filepath.Join(supervisor.Settings.RunDir(), "front.pid"))
}

// checkFront restarts the front-end when it exits, with bounded backoff.
func (supervisor *Supervisor) checkFront(now time.Time) {
	supervisor.mutex.Lock()
	cmd := supervisor.frontCmd
	startedAt := supervisor.frontAt
	supervisor.mutex.Unlock()
	if cmd == nil {
		return
	}
	if !supervisor.childExited(cmd) {
		return
	}
	supervisor.mutex.Lock()
	supervisor.frontCmd = nil
	supervisor.mutex.Unlock()
	supervisor.Log("front-end exited after %d s", int64(now.Sub(startedAt).Seconds()))
	if !supervisor.failure("front-end exited") {
		supervisor.SetState("failed", "too many failures; the last: the front-end exited")
		return
	}
	delay := supervisor.Settings.FrontDelays[0]
	if len(supervisor.fails) < len(supervisor.Settings.FrontDelays) {
		delay = supervisor.Settings.FrontDelays[len(supervisor.fails)-1]
	} else {
		delay = supervisor.Settings.FrontDelays[len(supervisor.Settings.FrontDelays)-1]
	}
	time.Sleep(time.Duration(delay) * time.Second)
	supervisor.startFront()
}

// checkEngine restarts the engine when rank 0 exits, with bounded backoff.
func (supervisor *Supervisor) checkEngine(now time.Time) {
	supervisor.mutex.Lock()
	cmd := supervisor.engineCmd
	startedAt := supervisor.engineAt
	supervisor.mutex.Unlock()
	if cmd == nil {
		return
	}
	if !supervisor.childExited(cmd) {
		return
	}
	supervisor.mutex.Lock()
	supervisor.engineCmd = nil
	supervisor.ready = false
	supervisor.mutex.Unlock()
	os.Remove(filepath.Join(supervisor.Settings.RunDir(), "rank0.pid"))
	supervisor.reapAll(10)
	supervisor.Log("rank 0 exited after %d s", int64(now.Sub(startedAt).Seconds()))
	if !supervisor.failure("rank 0 exited") {
		supervisor.SetState("failed", "too many failures; the last: rank 0 exited")
		return
	}
	delay := supervisor.Settings.EngineDelays[0]
	if len(supervisor.fails) < len(supervisor.Settings.EngineDelays) {
		delay = supervisor.Settings.EngineDelays[len(supervisor.fails)-1]
	} else {
		delay = supervisor.Settings.EngineDelays[len(supervisor.Settings.EngineDelays)-1]
	}
	supervisor.SetState("restarting", fmt.Sprintf("rank 0 exited; the next start at %s", time.Now().Add(time.Duration(delay)*time.Second).Format("15:04:05")))
	time.Sleep(time.Duration(delay) * time.Second)
}

// childExited reports whether a started child has exited, and reaps it when it has. A child that exited
// is a zombie until it is reaped, and a zombie still answers signal 0, so a liveness test that asks the
// process alone never sees the exit. Wait reaps it, and Wait returns as soon as the child exits.
func (supervisor *Supervisor) childExited(cmd *exec.Cmd) bool {
	exited := make(chan struct{})
	go func() {
		cmd.Wait()
		close(exited)
	}()
	select {
	case <-exited:
		return true
	case <-time.After(50 * time.Millisecond):
		return false
	}
}

// checkRing lends the ring to a job that waits once no request has come for YieldIdleS, and takes it
// back when requests wait.
func (supervisor *Supervisor) checkRing(now time.Time) {
	if supervisor.lent {
		return
	}
	wants, err := supervisor.Lock.Wants(false)
	if err != nil || len(wants) == 0 {
		return
	}
	idle := supervisor.engineCmd == nil || (supervisor.ready && int64(now.Sub(supervisor.readyAt).Seconds()) >= supervisor.Settings.YieldIdleS)
	if !idle {
		return
	}
	lines := make([]string, 0, len(wants))
	for _, want := range wants {
		lines = append(lines, want.Line)
	}
	supervisor.Log("lending the ring: no request for %d s; waiting for it: %s", supervisor.Settings.YieldIdleS, strings.Join(lines, "|"))
	supervisor.SetState("lending", "stopping the engine for a ring job")
	supervisor.stopEngine()
	supervisor.Lock.Release()
	supervisor.lent = true
	supervisor.lentAt = time.Now()
	supervisor.SetState("lent", fmt.Sprintf("to a ring job since %s; requests wait at the front-end, up to %d s each", supervisor.lentAt.Format("15:04:05"), supervisor.Settings.HoldS))
}

// checkWorkers stops rank 0 when a worker's rank is gone, so that the engine restarts.
func (supervisor *Supervisor) checkWorkers() {
	if !supervisor.ready {
		return
	}
	gone := make([]string, 0, len(supervisor.Settings.Workers))
	for index, worker := range supervisor.Settings.Workers {
		cmd := exec.Command("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", worker, "pgrep -x dsgb10_rank > /dev/null")
		if err := cmd.Run(); err != nil {
			gone = append(gone, fmt.Sprintf("rank %d (%s)", index+1, worker))
		}
	}
	if len(gone) == 0 {
		return
	}
	supervisor.Log("not running: %s; stopping rank 0 (SIGTERM) to restart the engine", strings.Join(gone, " "))
	supervisor.mutex.Lock()
	cmd := supervisor.engineCmd
	supervisor.mutex.Unlock()
	if cmd != nil && cmd.Process != nil {
		cmd.Process.Signal(syscall.SIGTERM)
	}
}

// failure counts a failure, and reports false once there were more than MaxFails within FailWindowS.
func (supervisor *Supervisor) failure(what string) bool {
	now := time.Now()
	kept := make([]time.Time, 0, len(supervisor.fails)+1)
	for _, at := range supervisor.fails {
		if now.Sub(at).Seconds() < float64(supervisor.Settings.FailWindowS) {
			kept = append(kept, at)
		}
	}
	supervisor.fails = append(kept, now)
	supervisor.restarts++
	line := fmt.Sprintf("%d restarts; the last at %s: %s\n", supervisor.restarts, now.Format("2006-01-02 15:04:05"), what)
	os.WriteFile(filepath.Join(supervisor.Settings.RunDir(), "restarts"), []byte(line), 0o644)
	supervisor.Log("failure: %s (%d in the last %d s)", what, len(supervisor.fails), supervisor.Settings.FailWindowS)
	return len(supervisor.fails) <= supervisor.Settings.MaxFails
}

// shutdown stops the line: the front-end first, so that /health stops answering 200 before the engine
// goes, then the engine.
func (supervisor *Supervisor) shutdown() {
	supervisor.SetState("stopping", "by the supervisor's stop")
	supervisor.stopFront()
	if !supervisor.lent {
		supervisor.stopEngine()
	}
	os.Remove(filepath.Join(supervisor.Settings.RunDir(), "engine.sock"))
	os.Remove(filepath.Join(supervisor.Settings.RunDir(), "front.admin"))
	supervisor.SetState("stopped", "by the supervisor's stop")
}

// reapAll stops every rank on every box, waiting up to waitS for them to exit by themselves.
func (supervisor *Supervisor) reapAll(waitS int64) {
	script := reapScript(waitS)
	done := make(chan struct{}, len(supervisor.Settings.Workers)+1)
	go func() {
		exec.Command("bash", "-c", script).Run()
		done <- struct{}{}
	}()
	for _, worker := range supervisor.Settings.Workers {
		go func(worker string) {
			exec.Command("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", worker, script).Run()
			done <- struct{}{}
		}(worker)
	}
	for index := 0; index <= len(supervisor.Settings.Workers); index++ {
		<-done
	}
}

// reapScript returns the shell text that stops every dsgb10_rank on one box.
func reapScript(waitS int64) string {
	return fmt.Sprintf("for i in $(seq %d); do pgrep -x dsgb10_rank > /dev/null || exit 0; sleep 0.5; done; pkill -x dsgb10_rank; for i in $(seq 20); do pgrep -x dsgb10_rank > /dev/null || exit 0; sleep 0.5; done; pkill -9 -x dsgb10_rank", 2*waitS)
}

// rotate moves a log aside: file -> file.1 -> ... -> file.Keep.
func (supervisor *Supervisor) rotate(path string) {
	info, err := os.Stat(path)
	if err != nil || info.Size() == 0 {
		return
	}
	for index := supervisor.Settings.Keep - 1; index >= 1; index-- {
		os.Rename(fmt.Sprintf("%s.%d", path, index), fmt.Sprintf("%s.%d", path, index+1))
	}
	os.Rename(path, path+".1")
}

// rotateLive rotates a log that a running process appends to, once it passes LogMiB: copy, truncate.
func (supervisor *Supervisor) rotateLive() {
	for _, name := range []string{"rank.0.log", "front.log", "service.log"} {
		path := filepath.Join(supervisor.Settings.LogDir(), name)
		info, err := os.Stat(path)
		if err != nil || info.Size() <= supervisor.Settings.LogMiB<<20 {
			continue
		}
		for index := supervisor.Settings.Keep - 1; index >= 1; index-- {
			os.Rename(fmt.Sprintf("%s.%d", path, index), fmt.Sprintf("%s.%d", path, index+1))
		}
		copyFile(path, path+".1")
		os.Truncate(path, 0)
	}
}

// copyFile copies source to target, and keeps the target's mode when it has one.
func copyFile(source string, target string) error {
	text, err := os.ReadFile(source)
	if err != nil {
		return fmt.Errorf("cannot read %s: %w", source, err)
	}
	info, err := os.Stat(source)
	mode := os.FileMode(0o755)
	if err == nil {
		mode = info.Mode().Perm()
	}
	temporary := target + ".new"
	if err := os.WriteFile(temporary, text, mode); err != nil {
		return fmt.Errorf("cannot write %s: %w", temporary, err)
	}
	if err := os.Rename(temporary, target); err != nil {
		return fmt.Errorf("cannot rename %s to %s: %w", temporary, target, err)
	}
	return nil
}

// memAvailableGiB returns MemAvailable of /proc/meminfo in GiB.
func memAvailableGiB() int64 {
	text, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(text), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kib, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return kib / 1048576
	}
	return 0
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

// homeDir returns the home directory of the user.
func homeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "/root"
	}
	return home
}
