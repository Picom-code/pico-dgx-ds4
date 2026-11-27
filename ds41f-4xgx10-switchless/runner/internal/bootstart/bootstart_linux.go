//go:build linux

package bootstart

import "syscall"

// detachedAttr makes the started supervisor its own session, so that it outlives the boot starter.
func detachedAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setsid: true}
}

// syscallSignalZero returns signal 0, which asks whether a process runs without sending it anything.
func syscallSignalZero() syscall.Signal {
	return syscall.Signal(0)
}
