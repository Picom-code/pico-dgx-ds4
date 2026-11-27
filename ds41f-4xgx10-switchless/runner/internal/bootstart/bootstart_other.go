//go:build !linux

package bootstart

import "syscall"

// detachedAttr returns nil on a host that has no setsid, so that the package builds and its tests run
// there. The line runs on a Linux box, where bootstart_linux.go supplies the real attribute.
func detachedAttr() *syscall.SysProcAttr {
	return nil
}

// syscallSignalZero returns signal 0, which asks whether a process runs without sending it anything.
func syscallSignalZero() syscall.Signal {
	return syscall.Signal(0)
}
