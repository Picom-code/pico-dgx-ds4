// Package ringlock holds the ring lock that every job on the ring takes, and the files beside it:
// ring.owner says whose the ring is, ring.wants holds one file per job that waits, and ring.reclaim is
// held by the service's supervisor while requests wait during a lend. engine/server/ringlock.sh holds
// the same files for the shell scripts, and this package holds them the same way, so that a Go job and
// a shell job queue in one order.
package ringlock

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Lock is the ring lock and the 3 files beside it.
type Lock struct {
	Path     string
	Owner    string
	WantsDir string
	Reclaim  string
	file     *os.File
}

// NewLock names the 3 files beside the lock file at path: ring.owner, ring.wants and ring.reclaim.
func NewLock(path string) *Lock {
	base := strings.TrimSuffix(path, ".lock")
	return &Lock{
		Path:     path,
		Owner:    base + ".owner",
		WantsDir: base + ".wants",
		Reclaim:  base + ".reclaim",
	}
}

// Take takes the lock without waiting, and reports whether it got it. The caller holds the lock until
// it calls Release or the process ends.
func (lock *Lock) Take() (bool, error) {
	file, err := os.OpenFile(lock.Path, os.O_RDWR|os.O_CREATE, 0o664)
	if err != nil {
		return false, fmt.Errorf("cannot open the ring lock %s: %w", lock.Path, err)
	}
	err = syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if err != nil {
		file.Close()
		if err == syscall.EWOULDBLOCK {
			return false, nil
		}
		return false, fmt.Errorf("cannot take the ring lock %s: %w", lock.Path, err)
	}
	lock.file = file
	return true, nil
}

// Release gives the lock back. A lock that is not held is not an error.
func (lock *Lock) Release() error {
	if lock.file == nil {
		return nil
	}
	err := syscall.Flock(int(lock.file.Fd()), syscall.LOCK_UN)
	closeErr := lock.file.Close()
	lock.file = nil
	if err != nil {
		return fmt.Errorf("cannot release the ring lock %s: %w", lock.Path, err)
	}
	return closeErr
}

// Held reports whether this process holds the lock.
func (lock *Lock) Held() bool {
	return lock.file != nil
}

// WriteOwner records whose the ring is, in the form the shell scripts write: the time, the pid, the
// who, and what it does.
func (lock *Lock) WriteOwner(who string) error {
	line := fmt.Sprintf("%s pid %d %s\n", time.Now().Format("2006-01-02 15:04:05"), os.Getpid(), who)
	err := os.WriteFile(lock.Owner, []byte(line), 0o664)
	if err != nil {
		return fmt.Errorf("cannot write %s: %w", lock.Owner, err)
	}
	return nil
}

// ReadOwner returns the owner line, or "" when no file holds one.
func (lock *Lock) ReadOwner() string {
	text, err := os.ReadFile(lock.Owner)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(text))
}

// Want is one job that waits for the ring.
type Want struct {
	Path  string
	Line  string
	Since int64
	Pid   int
}

// Wants returns the jobs that wait for the ring, oldest first. A want whose lock is free belongs to a
// job that is gone, and it is removed. With held set, the drivers' held wants are listed too.
func (lock *Lock) Wants(held bool) ([]Want, error) {
	entries, err := os.ReadDir(lock.WantsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("cannot read %s: %w", lock.WantsDir, err)
	}
	wants := make([]Want, 0, len(entries))
	for _, entry := range entries {
		name := entry.Name()
		if strings.HasSuffix(name, ".new") {
			continue
		}
		if strings.HasSuffix(name, ".held") && !held {
			continue
		}
		path := filepath.Join(lock.WantsDir, name)
		live, err := fileIsLocked(path)
		if err != nil || !live {
			os.Remove(path)
			continue
		}
		want := Want{Path: path}
		text, err := os.ReadFile(path)
		if err == nil {
			lines := strings.Split(strings.TrimRight(string(text), "\n"), "\n")
			want.Line = lines[0]
			if len(lines) > 1 {
				fields := strings.Fields(lines[1])
				if len(fields) > 0 {
					want.Since, _ = strconv.ParseInt(fields[0], 10, 64)
				}
				if len(fields) > 1 {
					want.Pid, _ = strconv.Atoi(fields[1])
				}
			}
		}
		wants = append(wants, want)
	}
	sort.Slice(wants, func(i int, j int) bool {
		if wants[i].Since != wants[j].Since {
			return wants[i].Since < wants[j].Since
		}
		return wants[i].Pid < wants[j].Pid
	})
	return wants, nil
}

// Reclaimed reports whether the service wants the ring back before any job: its supervisor holds the
// reclaim file's lock.
func (lock *Lock) Reclaimed() bool {
	live, err := fileIsLocked(lock.Reclaim)
	return err == nil && live
}

// fileIsLocked reports whether another process holds a shared or exclusive lock on path. A file that
// does not exist is not locked.
func fileIsLocked(path string) (bool, error) {
	file, err := os.OpenFile(path, os.O_RDWR, 0o664)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	defer file.Close()
	err = syscall.Flock(int(file.Fd()), syscall.LOCK_SH|syscall.LOCK_NB)
	if err != nil {
		if err == syscall.EWOULDBLOCK {
			return true, nil
		}
		return false, err
	}
	syscall.Flock(int(file.Fd()), syscall.LOCK_UN)
	return false, nil
}
