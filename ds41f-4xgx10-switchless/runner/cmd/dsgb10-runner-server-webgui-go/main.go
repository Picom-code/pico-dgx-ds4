// Command dsgb10-runner-server-webgui-go serves the line's state as one HTML page that refreshes itself:
// whether it serves, how long it has run, how often it restarted, the front-end's /health code, and the
// last 20 lines of the supervisor's log. It reads the files that the supervisor writes, so it needs no
// socket, and it holds no state of its own.
package main

import (
	"fmt"
	"html"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/fleetconf"
	"github.com/Picom-code/pico-dgx-ds4/ds41f-4xgx10-switchless/runner/internal/linestate"
)

func main() {
	listen := envOr("RUNNER_WEBGUI_LISTEN", "127.0.0.1:18895")
	for _, arg := range os.Args[1:] {
		switch arg {
		case "--help", "-h":
			usage()
			return
		default:
			if strings.HasPrefix(arg, "--listen=") {
				listen = strings.TrimPrefix(arg, "--listen=")
				continue
			}
			fmt.Fprintf(os.Stderr, "dsgb10-runner-server-webgui-go: unknown argument %s\n", arg)
			os.Exit(2)
		}
	}
	paths, err := readPaths()
	if err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-server-webgui-go: %v\n", err)
		os.Exit(2)
	}
	http.HandleFunc("/", func(writer http.ResponseWriter, request *http.Request) {
		state := paths.Read()
		writer.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(writer, renderPage(state))
	})
	http.HandleFunc("/state.json", func(writer http.ResponseWriter, request *http.Request) {
		state := paths.Read()
		text, err := state.JSON()
		if err != nil {
			http.Error(writer, err.Error(), http.StatusInternalServerError)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.Write(text)
	})
	fmt.Printf("dsgb10-runner-server-webgui-go: serving on http://%s/\n", listen)
	if err := http.ListenAndServe(listen, nil); err != nil {
		fmt.Fprintf(os.Stderr, "dsgb10-runner-server-webgui-go: %v\n", err)
		os.Exit(1)
	}
}

// readPaths reads the fleet file and returns the files of the line's service directory.
func readPaths() (linestate.Paths, error) {
	fleetPath := envOr("FLEET", "fleet/homelab.conf")
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
	return linestate.Paths{
		Home:  fleetconf.Local(serviceDir, 0, home),
		Addrs: splitAddrs(envOr("DSGB10_ADDR", "127.0.0.1")),
		Port:  envOr("DSGB10_PORT", "18893"),
	}, nil
}

// renderPage returns the page for one reading of the line's state. The page refreshes itself every 5 s.
func renderPage(state linestate.State) string {
	verdict := "not serving"
	colour := "#c0392b"
	if state.HealthOK {
		verdict = "serving"
		colour = "#27ae60"
	}
	var builder strings.Builder
	builder.WriteString("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n")
	builder.WriteString("<meta http-equiv=\"refresh\" content=\"5\">\n")
	builder.WriteString("<title>dsgb10 line</title>\n")
	builder.WriteString("<style>body{font-family:ui-monospace,Menlo,monospace;background:#111;color:#ddd;margin:2em}")
	builder.WriteString("h1{font-size:1.4em}table{border-collapse:collapse}td{padding:.15em 1em .15em 0;vertical-align:top}")
	builder.WriteString("pre{background:#1b1b1b;padding:1em;overflow-x:auto}.k{color:#888}</style>\n")
	builder.WriteString("</head><body>\n")
	builder.WriteString(fmt.Sprintf("<h1 style=\"color:%s\">%s</h1>\n", colour, html.EscapeString(verdict)))
	builder.WriteString("<table>\n")
	row := func(key string, value string) {
		builder.WriteString(fmt.Sprintf("<tr><td class=\"k\">%s</td><td>%s</td></tr>\n", html.EscapeString(key), html.EscapeString(value)))
	}
	row("state", state.State)
	row("since", state.Since.Format("2006-01-02 15:04:05"))
	row("uptime", formatUptime(state.UptimeS))
	row("what", state.What)
	row("restarts", state.Restarts)
	row("ring", state.RingOwner)
	row("front-end", fmt.Sprintf("%s, pid %d", strings.Join(state.URLs, " "), state.FrontPid))
	codes := make([]string, 0, len(state.HealthCodes))
	for _, code := range state.HealthCodes {
		codes = append(codes, fmt.Sprintf("%d", code))
	}
	row("/health", strings.Join(codes, " "))
	if state.Rank0Pid > 0 {
		row("rank 0", fmt.Sprintf("pid %d, up %ds", state.Rank0Pid, state.Rank0UptimeS))
	} else {
		row("rank 0", "not running")
	}
	row("MemAvailable", fmt.Sprintf("%d GiB", state.MemAvailGiB))
	row("read at", state.ReadAt.Format("15:04:05"))
	builder.WriteString("</table>\n")
	builder.WriteString(fmt.Sprintf("<h2>last %d lines of service.log</h2>\n<pre>", len(state.LogTail)))
	for _, line := range state.LogTail {
		builder.WriteString(html.EscapeString(line))
		builder.WriteString("\n")
	}
	builder.WriteString("</pre>\n</body></html>\n")
	return builder.String()
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
	fmt.Print(`dsgb10-runner-server-webgui-go: the line's state as one HTML page

  dsgb10-runner-server-webgui-go                    serve on 127.0.0.1:18895
  dsgb10-runner-server-webgui-go --listen=ADDR:PORT serve on another address

Paths: / (the page, refreshed every 5 s), /state.json (the same reading as JSON).
Settings, from the environment: FLEET (fleet/homelab.conf), DSGB10_ADDR, DSGB10_PORT,
RUNNER_WEBGUI_LISTEN (127.0.0.1:18895).
`)
}

// unused keeps the time import when the page is compiled out.
var _ = time.Now
