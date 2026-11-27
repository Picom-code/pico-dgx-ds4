// Package linestate reads the files that the supervisor writes, so that the CLI and the web GUI report
// the line without talking to the supervisor. The supervisor writes run/state, run/restarts and the
// logs; this package reads them and adds the front-end's /health code.
package linestate

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// State is the line as the supervisor last wrote it, plus what this package measured.
type State struct {
	State        string    `json:"state"`
	Since        time.Time `json:"since"`
	UptimeS      int64     `json:"uptime_s"`
	What         string    `json:"what"`
	Restarts     string    `json:"restarts"`
	RingOwner    string    `json:"ring_owner"`
	FrontPid     int       `json:"front_pid"`
	Rank0Pid     int       `json:"rank0_pid"`
	Rank0UptimeS int64     `json:"rank0_uptime_s"`
	URLs         []string  `json:"urls"`
	HealthCodes  []int     `json:"health_codes"`
	HealthOK     bool      `json:"health_ok"`
	MemAvailGiB  int64     `json:"mem_available_gib"`
	LogTail      []string  `json:"log_tail"`
	ReadAt       time.Time `json:"read_at"`
}

// Paths names the files of one service directory that this package reads.
type Paths struct {
	Home  string
	Addrs []string
	Port  string
}

// RunDir returns the service's run directory.
func (paths Paths) RunDir() string { return filepath.Join(paths.Home, "run") }

// LogDir returns the service's log directory.
func (paths Paths) LogDir() string { return filepath.Join(paths.Home, "logs") }

// ReadState reads run/state: "<state> <epoch> <what>", the words the supervisor last wrote.
func (paths Paths) ReadState() (string, time.Time, string, error) {
	text, err := os.ReadFile(filepath.Join(paths.RunDir(), "state"))
	if err != nil {
		return "", time.Time{}, "", err
	}
	fields := strings.SplitN(strings.TrimSpace(string(text)), " ", 3)
	if len(fields) < 2 {
		return "", time.Time{}, "", fmt.Errorf("run/state is not \"<state> <epoch> <what>\": %q", string(text))
	}
	epoch, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return "", time.Time{}, "", fmt.Errorf("run/state holds no epoch: %q", fields[1])
	}
	what := ""
	if len(fields) > 2 {
		what = fields[2]
	}
	return fields[0], time.Unix(epoch, 0), what, nil
}

// ReadRestarts reads run/restarts: "<n> restarts; the last at <time>: <what>".
func (paths Paths) ReadRestarts() string {
	text, err := os.ReadFile(filepath.Join(paths.RunDir(), "restarts"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(text))
}

// ReadPid reads a pid file, and returns 0 when the file holds no pid.
func (paths Paths) ReadPid(name string) int {
	text, err := os.ReadFile(filepath.Join(paths.RunDir(), name))
	if err != nil {
		return 0
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(text)))
	if err != nil {
		return 0
	}
	return pid
}

// ProcessUptimeS returns how long the process has run, in seconds, and 0 when the process does not run.
// It reads /proc, which a Linux box has. A host without /proc, such as the Mac the runner is developed
// on, reports 0, and the caller prints the start time instead.
func ProcessUptimeS(pid int) int64 {
	if pid <= 0 {
		return 0
	}
	stat, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(stat))
	if len(fields) < 22 {
		return 0
	}
	startTicks, err := strconv.ParseInt(fields[21], 10, 64)
	if err != nil {
		return 0
	}
	uptimeText, err := os.ReadFile("/proc/uptime")
	if err != nil {
		return 0
	}
	uptimeFields := strings.Fields(string(uptimeText))
	if len(uptimeFields) == 0 {
		return 0
	}
	uptimeS, err := strconv.ParseFloat(uptimeFields[0], 64)
	if err != nil {
		return 0
	}
	const clockTicksPerS = 100
	return int64(uptimeS) - startTicks/clockTicksPerS
}

// ProcessRuns reports whether the process runs. A process that exited but was not reaped still answers
// signal 0, so the caller that started it must reap it instead of asking this.
func ProcessRuns(pid int) bool {
	if pid <= 0 {
		return false
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return process.Signal(syscall.Signal(0)) == nil
}

// MemAvailableGiB returns MemAvailable of /proc/meminfo in GiB, and 0 when the file is not readable.
func MemAvailableGiB() int64 {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
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

// HealthCode asks one address for /health and returns the HTTP code, or 0 when it does not answer.
func HealthCode(addr string, port string) int {
	client := &http.Client{Timeout: 3 * time.Second}
	response, err := client.Get("http://" + addr + ":" + port + "/health")
	if err != nil {
		return 0
	}
	defer response.Body.Close()
	return response.StatusCode
}

// TailLog returns the last count lines of a log file, oldest first. A log with fewer lines returns the
// lines it has.
func TailLog(path string, count int) []string {
	text, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	lines := strings.Split(strings.TrimRight(string(text), "\n"), "\n")
	if len(lines) > count {
		lines = lines[len(lines)-count:]
	}
	if len(lines) == 1 && lines[0] == "" {
		return nil
	}
	return lines
}

// Read collects the line's state from the files and from the front-end's /health.
func (paths Paths) Read() State {
	state := State{ReadAt: time.Now()}
	stateName, since, what, err := paths.ReadState()
	if err == nil {
		state.State = stateName
		state.Since = since
		state.What = what
		state.UptimeS = int64(time.Since(since).Seconds())
	} else {
		state.State = "not running"
	}
	state.Restarts = paths.ReadRestarts()
	state.RingOwner = readOwnerBeside(paths)
	state.FrontPid = paths.ReadPid("front.pid")
	state.Rank0Pid = paths.ReadPid("rank0.pid")
	state.Rank0UptimeS = ProcessUptimeS(state.Rank0Pid)
	state.MemAvailGiB = MemAvailableGiB()
	state.HealthOK = true
	for _, addr := range paths.Addrs {
		code := HealthCode(addr, paths.Port)
		state.HealthCodes = append(state.HealthCodes, code)
		state.URLs = append(state.URLs, "http://"+addr+":"+paths.Port+"/v1")
		if code != http.StatusOK {
			state.HealthOK = false
		}
	}
	state.LogTail = TailLog(filepath.Join(paths.LogDir(), "service.log"), 20)
	return state
}

// readOwnerBeside returns the ring owner line, read from the owner file in the service's home.
func readOwnerBeside(paths Paths) string {
	text, err := os.ReadFile(filepath.Join(paths.Home, "ring.owner"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(text))
}

// JSON returns the state as one JSON object.
func (state State) JSON() ([]byte, error) {
	return json.MarshalIndent(state, "", "  ")
}
