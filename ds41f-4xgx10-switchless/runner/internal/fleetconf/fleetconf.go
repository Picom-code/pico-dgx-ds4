// Package fleetconf reads a fleet file: the KEY=value lines that name the boxes of the ring, how they
// are addressed, and where each keeps its files. engine/server/fleet.sh reads the same file for the
// shell scripts, and engine/comm/fleet.cpp reads it for the engine. This package reads it for the Go
// runner, so that the runner and the scripts agree on one file.
package fleetconf

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Fleet is one fleet file, read into its keys.
type Fleet struct {
	Path string
	Keys map[string]string
}

// ReadFleet reads the fleet file at path. A line is a KEY=value pair or a comment that starts with #.
// A value holds no spaces and no quotes: letters, digits and . _ - / ~ : , @ + % { }.
func ReadFleet(path string) (*Fleet, error) {
	text, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read the fleet file %s: %w", path, err)
	}
	fleet := &Fleet{Path: path, Keys: make(map[string]string)}
	for lineNumber, line := range strings.Split(string(text), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found {
			return nil, fmt.Errorf("%s line %d: not a KEY=value line: %q", path, lineNumber+1, line)
		}
		key = strings.TrimSpace(key)
		if key == "" {
			return nil, fmt.Errorf("%s line %d: no key before =", path, lineNumber+1)
		}
		fleet.Keys[key] = strings.TrimSpace(value)
	}
	return fleet, nil
}

// Get returns the value of key, and whether the file names it.
func (fleet *Fleet) Get(key string) (string, bool) {
	value, found := fleet.Keys[key]
	return value, found
}

// MustGet returns the value of key, or an error that names the file and the key.
func (fleet *Fleet) MustGet(key string) (string, error) {
	value, found := fleet.Keys[key]
	if !found {
		return "", fmt.Errorf("the fleet file %s names no %s", fleet.Path, key)
	}
	return value, nil
}

// Local expands a fleet path for the box that reads it. A leading ~/ is the home directory of the user
// on that box, and {rank} is the rank number.
func Local(path string, rank int, home string) string {
	expanded := strings.ReplaceAll(path, "{rank}", strconv.Itoa(rank))
	if strings.HasPrefix(expanded, "~/") {
		return filepath.Join(home, expanded[2:])
	}
	return expanded
}

// Workers returns the SSH names of ranks 1 to 3, in rank order, read from RANK<r>_SSH. A fleet that
// names no RANK1_SSH is a one-box fleet, such as a test's own, and it has no workers.
func (fleet *Fleet) Workers() ([]string, error) {
	if _, found := fleet.Get("RANK1_SSH"); !found {
		return nil, nil
	}
	workers := make([]string, 0, 3)
	for rank := 1; rank <= 3; rank++ {
		name, err := fleet.MustGet("RANK" + strconv.Itoa(rank) + "_SSH")
		if err != nil {
			return nil, err
		}
		workers = append(workers, name)
	}
	return workers, nil
}

// Rank0Name returns the name of rank 0, from RANK0_NAME, or "rank 0" when the file names none.
func (fleet *Fleet) Rank0Name() string {
	if name, found := fleet.Get("RANK0_NAME"); found && name != "" {
		return name
	}
	return "rank 0"
}

// ServiceDir returns the service's directory on rank 0, from SERVICE_DIR.
func (fleet *Fleet) ServiceDir() (string, error) {
	return fleet.MustGet("SERVICE_DIR")
}

// RingLock returns the ring lock's path on rank 0, from RING_LOCK.
func (fleet *Fleet) RingLock() (string, error) {
	return fleet.MustGet("RING_LOCK")
}

// ModelFiles returns the model files of rank, from RANK<r>_MODEL: the files whose page cache the start
// drops before the engine loads them.
func (fleet *Fleet) ModelFiles(rank int) []string {
	value, found := fleet.Get("RANK" + strconv.Itoa(rank) + "_MODEL")
	if !found || value == "" {
		return nil
	}
	return strings.Fields(value)
}

// SpillDir returns the engine's spill directory of rank, from RANK<r>_SPILL.
func (fleet *Fleet) SpillDir(rank int) (string, bool) {
	return fleet.Get("RANK" + strconv.Itoa(rank) + "_SPILL")
}

// Keys returns every key of the file, sorted, so that a report of the file is stable.
func (fleet *Fleet) KeysSorted() []string {
	keys := make([]string, 0, len(fleet.Keys))
	for key := range fleet.Keys {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
