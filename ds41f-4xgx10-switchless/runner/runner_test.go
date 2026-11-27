// The integration tests of the runner. They run the real binaries against a stub engine and a stub
// front-end, in a service directory of their own under the test's temporary folder, so that they touch
// no line and no ring. A test states what an operator sees, not how the code is arranged.
package runner_test

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// stubFront is a front-end that answers /health with the codes a test gives it, in order, and repeats
// the last code after that. It answers /v1/models with 200 always.
type stubFront struct {
	server   *httptest.Server
	codes    []int
	answered int64
}

// newStubFront starts a stub front-end that answers the codes in order.
func newStubFront(t *testing.T, codes ...int) *stubFront {
	t.Helper()
	stub := &stubFront{codes: codes}
	stub.server = httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/health" {
			writer.WriteHeader(http.StatusOK)
			return
		}
		index := int(atomic.AddInt64(&stub.answered, 1)) - 1
		if index >= len(stub.codes) {
			index = len(stub.codes) - 1
		}
		writer.WriteHeader(stub.codes[index])
	}))
	t.Cleanup(stub.server.Close)
	return stub
}

// port returns the port the stub listens on.
func (stub *stubFront) port() string {
	_, port, _ := strings.Cut(strings.TrimPrefix(stub.server.URL, "http://"), ":")
	return port
}

// serviceDir is a service directory of the test's own: the fleet file, the key, and the directories the
// supervisor writes.
type serviceDir struct {
	root  string
	key   string
	fleet string
}

// newServiceDir makes a service directory under the test's temporary folder.
func newServiceDir(t *testing.T) *serviceDir {
	t.Helper()
	// A Unix socket path may hold at most 104 bytes on macOS, and the engine's socket sits in run/. The
	// test's temporary folder is named after the test, so a long test name would push the socket path past
	// that limit and the stub engine's bind would fail. A short folder under the system temporary folder
	// keeps the path short.
	root, err := os.MkdirTemp("", "m7")
	if err != nil {
		t.Fatalf("cannot make the service directory: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(root) })
	dir := &serviceDir{
		root:  root,
		key:   filepath.Join(root, "key"),
		fleet: filepath.Join(root, "fleet.conf"),
	}
	if err := os.WriteFile(dir.key, []byte("test-key\n"), 0o600); err != nil {
		t.Fatalf("cannot write the key file: %v", err)
	}
	fleet := fmt.Sprintf("SERVICE_DIR=%s\nRING_LOCK=%s\n", root, filepath.Join(root, "ring.lock"))
	if err := os.WriteFile(dir.fleet, []byte(fleet), 0o644); err != nil {
		t.Fatalf("cannot write the fleet file: %v", err)
	}
	for _, name := range []string{"run", "logs", "bin"} {
		if err := os.MkdirAll(filepath.Join(root, name), 0o755); err != nil {
			t.Fatalf("cannot make %s: %v", name, err)
		}
	}
	return dir
}

// writeState writes run/state as the supervisor writes it.
func (dir *serviceDir) writeState(t *testing.T, state string, since time.Time, what string) {
	t.Helper()
	line := fmt.Sprintf("%s %d %s\n", state, since.Unix(), what)
	if err := os.WriteFile(filepath.Join(dir.root, "run", "state"), []byte(line), 0o644); err != nil {
		t.Fatalf("cannot write run/state: %v", err)
	}
}

// writeRestarts writes run/restarts as the supervisor writes it.
func (dir *serviceDir) writeRestarts(t *testing.T, text string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir.root, "run", "restarts"), []byte(text+"\n"), 0o644); err != nil {
		t.Fatalf("cannot write run/restarts: %v", err)
	}
}

// writeLog writes the supervisor's log with count lines.
func (dir *serviceDir) writeLog(t *testing.T, count int) {
	t.Helper()
	var builder strings.Builder
	for index := 1; index <= count; index++ {
		builder.WriteString(fmt.Sprintf("2026-11-27 10:00:00 log line %d\n", index))
	}
	if err := os.WriteFile(filepath.Join(dir.root, "logs", "service.log"), []byte(builder.String()), 0o644); err != nil {
		t.Fatalf("cannot write service.log: %v", err)
	}
}

// buildBinary builds one command of the module into the test's temporary folder, and returns its path.
func buildBinary(t *testing.T, name string) string {
	t.Helper()
	target := filepath.Join(t.TempDir(), name)
	cmd := exec.Command("go", "build", "-o", target, "./cmd/"+name)
	cmd.Dir = moduleRoot(t)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("cannot build %s: %v\n%s", name, err, output)
	}
	return target
}

// moduleRoot returns the module's directory, which holds go.mod.
func moduleRoot(t *testing.T) string {
	t.Helper()
	directory, err := os.Getwd()
	if err != nil {
		t.Fatalf("cannot read the working directory: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(directory, "go.mod")); err == nil {
			return directory
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			t.Fatalf("no go.mod above %s", directory)
		}
		directory = parent
	}
}

// runCLI runs the CLI against the service directory, and returns its output and its exit code.
func runCLI(t *testing.T, dir *serviceDir, port string, extra ...string) (string, int) {
	t.Helper()
	binary := buildBinary(t, "dsgb10-runner-server-cli-go")
	args := append([]string{}, extra...)
	cmd := exec.Command(binary, args...)
	cmd.Env = append(os.Environ(),
		"FLEET="+dir.fleet,
		"DSGB10_ADDR=127.0.0.1",
		"DSGB10_PORT="+port,
	)
	output, err := cmd.CombinedOutput()
	code := 0
	if exitErr, ok := err.(*exec.ExitError); ok {
		code = exitErr.ExitCode()
	} else if err != nil {
		t.Fatalf("cannot run the CLI: %v", err)
	}
	return string(output), code
}

// TestCLIReportsServingLine proves the CLI reports a serving line: the state, the uptime, the restart
// count, the /health code, and the last 20 lines of the log.
func TestCLIReportsServingLine(t *testing.T) {
	dir := newServiceDir(t)
	stub := newStubFront(t, http.StatusOK)
	dir.writeState(t, "serving", time.Now().Add(-90*time.Second), "http://127.0.0.1:"+stub.port()+"/v1; engine up")
	dir.writeRestarts(t, "2 restarts; the last at 2026-11-27 10:00:00: rank 0 exited")
	dir.writeLog(t, 30)
	output, code := runCLI(t, dir, stub.port())
	if code != 0 {
		t.Fatalf("the CLI exited %d for a serving line; output:\n%s", code, output)
	}
	for _, want := range []string{"service: serving", "uptime: 1m", "2 restarts", "/health 200", "the engine is connected"} {
		if !strings.Contains(output, want) {
			t.Errorf("the CLI output does not hold %q; output:\n%s", want, output)
		}
	}
	tail := countLinesWithPrefix(output, "    2026-11-27 10:00:00 log line ")
	if tail != 20 {
		t.Errorf("the CLI printed %d log lines, want 20; output:\n%s", tail, output)
	}
	if !strings.Contains(output, "log line 30") {
		t.Errorf("the CLI's log tail does not end at the newest line; output:\n%s", output)
	}
	if strings.Contains(output, "log line 10") {
		t.Errorf("the CLI's log tail reaches past the last 20 lines; output:\n%s", output)
	}
}

// TestCLIReportsNotServingLine proves the CLI exits 1 and says so when the front-end does not answer
// 200, and when no line runs at all.
func TestCLIReportsNotServingLine(t *testing.T) {
	dir := newServiceDir(t)
	stub := newStubFront(t, http.StatusServiceUnavailable)
	dir.writeState(t, "serving", time.Now(), "engine up")
	output, code := runCLI(t, dir, stub.port())
	if code != 1 {
		t.Errorf("the CLI exited %d for a line whose /health is 503, want 1; output:\n%s", code, output)
	}
	if !strings.Contains(output, "/health 503") || !strings.Contains(output, "not serving") {
		t.Errorf("the CLI does not report the 503; output:\n%s", output)
	}
	empty := newServiceDir(t)
	output, code = runCLI(t, empty, stub.port())
	if code != 1 {
		t.Errorf("the CLI exited %d for a line that does not run, want 1; output:\n%s", code, output)
	}
	if !strings.Contains(output, "service: not running") {
		t.Errorf("the CLI does not report a line that does not run; output:\n%s", output)
	}
}

// TestCLIReportsHealthRecovering proves the CLI reports the /health code it reads now: a front-end that
// answers 500 three times and then 200 is reported as failing, then as serving.
func TestCLIReportsHealthRecovering(t *testing.T) {
	dir := newServiceDir(t)
	stub := newStubFront(t, http.StatusInternalServerError, http.StatusInternalServerError, http.StatusInternalServerError, http.StatusOK)
	dir.writeState(t, "serving", time.Now(), "engine up")
	for attempt := 1; attempt <= 3; attempt++ {
		output, code := runCLI(t, dir, stub.port())
		if code != 1 {
			t.Fatalf("attempt %d: the CLI exited %d while /health answers 500, want 1; output:\n%s", attempt, code, output)
		}
		if !strings.Contains(output, "/health 500") {
			t.Fatalf("attempt %d: the CLI does not report the 500; output:\n%s", attempt, output)
		}
	}
	output, code := runCLI(t, dir, stub.port())
	if code != 0 {
		t.Fatalf("the CLI exited %d after /health recovered to 200, want 0; output:\n%s", code, output)
	}
	if !strings.Contains(output, "/health 200") {
		t.Errorf("the CLI does not report the recovered 200; output:\n%s", output)
	}
}

// TestCLIJSON proves the CLI's --json output holds every field the web GUI shows.
func TestCLIJSON(t *testing.T) {
	dir := newServiceDir(t)
	stub := newStubFront(t, http.StatusOK)
	dir.writeState(t, "serving", time.Now().Add(-30*time.Second), "engine up")
	dir.writeRestarts(t, "0 restarts")
	dir.writeLog(t, 3)
	output, code := runCLI(t, dir, stub.port(), "--json")
	if code != 0 {
		t.Fatalf("the CLI exited %d for --json on a serving line; output:\n%s", code, output)
	}
	var state map[string]any
	if err := json.Unmarshal([]byte(output), &state); err != nil {
		t.Fatalf("the CLI's --json output is not JSON: %v\n%s", err, output)
	}
	for _, key := range []string{"state", "since", "uptime_s", "restarts", "health_codes", "health_ok", "log_tail"} {
		if _, found := state[key]; !found {
			t.Errorf("the CLI's --json output has no %q; output:\n%s", key, output)
		}
	}
	if state["state"] != "serving" {
		t.Errorf("the CLI's --json state is %v, want serving", state["state"])
	}
}

// TestWebGUIServesTheSameReading proves the web GUI's page and its /state.json report the same line as
// the CLI: the state, the uptime, the restart count, the /health code, and the log tail.
func TestWebGUIServesTheSameReading(t *testing.T) {
	dir := newServiceDir(t)
	stub := newStubFront(t, http.StatusOK)
	dir.writeState(t, "serving", time.Now().Add(-125*time.Second), "engine up")
	dir.writeRestarts(t, "1 restarts; the last at 2026-11-27 10:00:00: rank 0 exited")
	dir.writeLog(t, 25)
	binary := buildBinary(t, "dsgb10-runner-server-webgui-go")
	listen := "127.0.0.1:" + freePort(t)
	cmd := exec.Command(binary, "--listen="+listen)
	cmd.Env = append(os.Environ(),
		"FLEET="+dir.fleet,
		"DSGB10_ADDR=127.0.0.1",
		"DSGB10_PORT="+stub.port(),
	)
	if err := cmd.Start(); err != nil {
		t.Fatalf("cannot start the web GUI: %v", err)
	}
	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
	})
	waitForHTTP(t, "http://"+listen+"/state.json")
	response, err := http.Get("http://" + listen + "/state.json")
	if err != nil {
		t.Fatalf("cannot read /state.json: %v", err)
	}
	defer response.Body.Close()
	var state map[string]any
	if err := json.NewDecoder(response.Body).Decode(&state); err != nil {
		t.Fatalf("/state.json is not JSON: %v", err)
	}
	if state["state"] != "serving" {
		t.Errorf("/state.json state is %v, want serving", state["state"])
	}
	if uptime, ok := state["uptime_s"].(float64); !ok || uptime < 120 {
		t.Errorf("/state.json uptime_s is %v, want at least 120", state["uptime_s"])
	}
	if restarts, _ := state["restarts"].(string); !strings.Contains(restarts, "1 restarts") {
		t.Errorf("/state.json restarts is %v, want 1 restarts", state["restarts"])
	}
	codes, ok := state["health_codes"].([]any)
	if !ok || len(codes) != 1 || codes[0].(float64) != 200 {
		t.Errorf("/state.json health_codes is %v, want [200]", state["health_codes"])
	}
	tail, ok := state["log_tail"].([]any)
	if !ok || len(tail) != 20 {
		t.Errorf("/state.json log_tail holds %d lines, want 20", len(tail))
	}
	page, err := http.Get("http://" + listen + "/")
	if err != nil {
		t.Fatalf("cannot read the page: %v", err)
	}
	defer page.Body.Close()
	body := make([]byte, 1<<20)
	count, _ := page.Body.Read(body)
	text := string(body[:count])
	for _, want := range []string{"serving", "uptime", "restarts", "/health", "200", "log line 25"} {
		if !strings.Contains(text, want) {
			t.Errorf("the page does not hold %q", want)
		}
	}
}

// TestSupervisorStartsAndStops proves the supervisor starts the front-end and the engine, reaches
// serving, and stops both when it is asked to.
func TestSupervisorStartsAndStops(t *testing.T) {
	dir := newServiceDir(t)
	port := freePort(t)
	writeStubFront(t, dir, port)
	writeStubEngine(t, dir)
	binary := buildBinary(t, "dsgb10-runner-server-go")
	cmd := exec.Command(binary)
	cmd.Env = append(os.Environ(),
		"FLEET="+dir.fleet,
		"DSGB10_ADDR=127.0.0.1",
		"DSGB10_PORT="+port,
		"DSGB10_KEY_FILE="+dir.key,
		"DSGB10_BIN="+filepath.Join(dir.root, "bin", "dsgb10_rank"),
		"DSGB10_FRONT_BIN="+filepath.Join(dir.root, "bin", "dsfront"),
		"MIN_AVAIL_GIB=0",
	)
	if err := cmd.Start(); err != nil {
		t.Fatalf("cannot start the supervisor: %v", err)
	}
	stopped := false
	t.Cleanup(func() {
		if !stopped {
			cmd.Process.Kill()
			cmd.Wait()
		}
	})
	statePath := filepath.Join(dir.root, "run", "state")
	waitForState(t, statePath, "serving", 30*time.Second)
	response, err := http.Get("http://127.0.0.1:" + port + "/health")
	if err != nil {
		t.Fatalf("the front-end does not answer /health: %v", err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Errorf("/health answers %d, want 200", response.StatusCode)
	}
	cmd.Process.Signal(os.Interrupt)
	done := make(chan struct{})
	go func() {
		cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
		stopped = true
	case <-time.After(30 * time.Second):
		t.Fatalf("the supervisor did not stop within 30 s of SIGINT")
	}
	waitForState(t, statePath, "stopped", 5*time.Second)
}

// TestSupervisorRestartsTheEngine proves the restarter: the supervisor starts the engine again after
// rank 0 is killed, and counts the restart.
func TestSupervisorRestartsTheEngine(t *testing.T) {
	dir := newServiceDir(t)
	port := freePort(t)
	writeStubFront(t, dir, port)
	writeStubEngine(t, dir)
	binary := buildBinary(t, "dsgb10-runner-server-go")
	cmd := exec.Command(binary)
	cmd.Env = append(os.Environ(),
		"FLEET="+dir.fleet,
		"DSGB10_ADDR=127.0.0.1",
		"DSGB10_PORT="+port,
		"DSGB10_KEY_FILE="+dir.key,
		"DSGB10_BIN="+filepath.Join(dir.root, "bin", "dsgb10_rank"),
		"DSGB10_FRONT_BIN="+filepath.Join(dir.root, "bin", "dsfront"),
		"MIN_AVAIL_GIB=0",
	)
	if err := cmd.Start(); err != nil {
		t.Fatalf("cannot start the supervisor: %v", err)
	}
	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
	})
	statePath := filepath.Join(dir.root, "run", "state")
	waitForState(t, statePath, "serving", 30*time.Second)
	pidText, err := os.ReadFile(filepath.Join(dir.root, "run", "rank0.pid"))
	if err != nil {
		t.Fatalf("the supervisor wrote no rank0.pid: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(pidText)))
	if err != nil {
		t.Fatalf("rank0.pid holds no pid: %v", err)
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		t.Fatalf("cannot find rank 0: %v", err)
	}
	if err := process.Kill(); err != nil {
		t.Fatalf("cannot kill rank 0: %v", err)
	}
	waitForFileContains(t, filepath.Join(dir.root, "run", "restarts"), "1 restarts", 60*time.Second)
	waitForState(t, statePath, "serving", 60*time.Second)
	newPidText, err := os.ReadFile(filepath.Join(dir.root, "run", "rank0.pid"))
	if err != nil {
		t.Fatalf("the supervisor wrote no rank0.pid after the restart: %v", err)
	}
	newPid, err := strconv.Atoi(strings.TrimSpace(string(newPidText)))
	if err != nil {
		t.Fatalf("rank0.pid holds no pid after the restart: %v", err)
	}
	if newPid == pid {
		t.Errorf("rank 0 still runs as pid %d, so it did not restart", pid)
	}
}

// TestBootStarterStartsTheSupervisor proves the boot starter starts the supervisor, which then serves.
func TestBootStarterStartsTheSupervisor(t *testing.T) {
	dir := newServiceDir(t)
	port := freePort(t)
	writeStubFront(t, dir, port)
	writeStubEngine(t, dir)
	serverBinary := buildBinary(t, "dsgb10-runner-server-go")
	bootBinary := buildBinary(t, "dsgb10-runner-bootstart-go")
	cmd := exec.Command(bootBinary)
	cmd.Env = append(os.Environ(),
		"FLEET="+dir.fleet,
		"DSGB10_ADDR=127.0.0.1",
		"DSGB10_PORT="+port,
		"DSGB10_KEY_FILE="+dir.key,
		"DSGB10_BIN="+filepath.Join(dir.root, "bin", "dsgb10_rank"),
		"DSGB10_FRONT_BIN="+filepath.Join(dir.root, "bin", "dsfront"),
		"DSGB10_SERVER_BIN="+serverBinary,
		"MIN_AVAIL_GIB=0",
		"BOOT_WAIT_S=1",
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("the boot starter exited with an error: %v\n%s", err, output)
	}
	// The boot starter returns as soon as it has started the supervisor, and the supervisor writes its
	// pid file after that, so the test waits for the file instead of reading it at once.
	pidPath := filepath.Join(dir.root, "run", "supervisor.pid")
	waitForFile(t, pidPath, 20*time.Second)
	pidText, err := os.ReadFile(pidPath)
	if err != nil {
		t.Fatalf("the boot starter started no supervisor: %v\n%s", err, output)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(pidText)))
	if err != nil {
		t.Fatalf("supervisor.pid holds no pid: %v", err)
	}
	t.Cleanup(func() {
		if process, err := os.FindProcess(pid); err == nil {
			process.Signal(os.Interrupt)
			time.Sleep(3 * time.Second)
			process.Kill()
		}
	})
	waitForState(t, filepath.Join(dir.root, "run", "state"), "serving", 30*time.Second)
}

// writeStubFront writes a stub front-end into the service directory's bin folder. It answers /health
// with 200 and /v1/models with 200, and it stops on SIGTERM.
func writeStubFront(t *testing.T, dir *serviceDir, port string) {
	t.Helper()
	script := fmt.Sprintf(`#!/bin/sh
# A stub front-end for the runner's tests: /health 200, /v1/models 200, until SIGTERM.
exec python3 -c '
import http.server, signal, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"{\"ok\":true}" if self.path == "/health" else b"{\"data\":[]}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
http.server.HTTPServer(("127.0.0.1", %s), H).serve_forever()
'
`, port)
	path := filepath.Join(dir.root, "bin", "dsfront")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("cannot write the stub front-end: %v", err)
	}
}

// writeStubEngine writes a stub engine into the service directory's bin folder. It makes the socket it
// is given and stays up until SIGTERM.
func writeStubEngine(t *testing.T, dir *serviceDir) {
	t.Helper()
	socket := filepath.Join(dir.root, "run", "engine.sock")
	script := fmt.Sprintf(`#!/bin/sh
# A stub engine for the runner's tests: it makes its socket and stays up until SIGTERM.
sock=%s
for arg in "$@"; do
  case "$prev" in --socket) sock=$arg ;; esac
  prev=$arg
done
exec python3 -c '
import os, signal, socket, sys, time
sock = sys.argv[1]
try: os.unlink(sock)
except OSError: pass
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sock)
server.listen(8)
signal.signal(signal.SIGTERM, lambda *args: (os.unlink(sock) if os.path.exists(sock) else None, sys.exit(0)))
while True: time.sleep(0.2)
' "$sock"
`, socket)
	path := filepath.Join(dir.root, "bin", "dsgb10_rank")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("cannot write the stub engine: %v", err)
	}
}

// netListen listens on a free port of the loopback address.
func netListen() (net.Listener, error) {
	return net.Listen("tcp", "127.0.0.1:0")
}

// freePort returns a port that nothing listens on now.
func freePort(t *testing.T) string {
	t.Helper()
	listener, err := netListen()
	if err != nil {
		t.Fatalf("cannot find a free port: %v", err)
	}
	defer listener.Close()
	_, port, _ := strings.Cut(listener.Addr().String(), ":")
	return port
}

// waitForState waits until run/state names the state, and fails the test when it does not in time.
func waitForState(t *testing.T, path string, state string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		text, err := os.ReadFile(path)
		if err == nil {
			fields := strings.Fields(string(text))
			if len(fields) > 0 && fields[0] == state {
				return
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	text, _ := os.ReadFile(path)
	t.Fatalf("run/state did not reach %q within %s; it holds %q", state, timeout, strings.TrimSpace(string(text)))
}

// waitForFile waits until a file exists, and fails the test when it does not in time.
func waitForFile(t *testing.T, path string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("%s did not appear within %s", path, timeout)
}

// waitForFileContains waits until a file holds the text, and fails the test when it does not in time.
func waitForFileContains(t *testing.T, path string, want string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		text, err := os.ReadFile(path)
		if err == nil && strings.Contains(string(text), want) {
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	text, _ := os.ReadFile(path)
	t.Fatalf("%s did not hold %q within %s; it holds %q", path, want, timeout, strings.TrimSpace(string(text)))
}

// waitForHTTP waits until the URL answers, and fails the test when it does not in time.
func waitForHTTP(t *testing.T, url string) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		response, err := http.Get(url)
		if err == nil {
			response.Body.Close()
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("%s did not answer within 20 s", url)
}

// countLinesWithPrefix counts the output's lines that start with prefix.
func countLinesWithPrefix(output string, prefix string) int {
	count := 0
	for _, line := range strings.Split(output, "\n") {
		if strings.HasPrefix(line, prefix) {
			count++
		}
	}
	return count
}
