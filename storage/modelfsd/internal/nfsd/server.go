package nfsd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/netip"
	"runtime/debug"
	"sync"
	"time"
)

// The server: one listener, one goroutine per connection, everything bounded.
//
// Both programs are served on the same listener. MOUNT normally lives on a port of its
// own found through rpcbind; we generate the consumer's mount line ourselves, so it can
// say mountport= and there is no reason for a second socket or a second daemon.

// Defaults. Each exists to bound something a LAN client would otherwise control: how
// many connections it can open, how much work it can have in flight on each, how large
// a message it can make the server hold, and how long it can hold a socket idle.
const (
	// DefaultPort is the port NFS clients expect. Nothing here binds it; the caller
	// supplies the listener, so the appliance decides which address it appears on.
	DefaultPort = 2049

	defaultMaxConnections = 128
	defaultMaxInFlight    = 16
	defaultMaxMessageSize = 1 << 20
	defaultIdleTimeout    = 5 * time.Minute
	defaultWriteTimeout   = 60 * time.Second
)

// ErrNoBackend means a server was configured without an index to serve from.
var ErrNoBackend = errors.New("nfsd: no backend configured")

// Config describes one server.
type Config struct {
	// Backend is the index to serve. Required.
	Backend Backend
	// Exports are the share names EXPORT advertises, each filtered by the caller's
	// grant. An empty list means EXPORT returns nothing, which is not the same as
	// refusing a mount: MNT consults the backend, not this list.
	Exports []string
	// MaxConnections bounds concurrent client connections. A connection beyond the
	// limit is closed immediately rather than queued: a queued connection looks alive
	// to the client while being served by nobody, and a client that is refused
	// retries.
	MaxConnections int
	// MaxInFlight bounds concurrent requests on one connection. Together with the
	// cap on a single READ, this is what bounds the server's memory under load.
	MaxInFlight int
	// MaxMessageSize bounds one reassembled RPC record. A larger record is not an
	// error we can reply to — the transaction id is inside it — so the connection is
	// dropped.
	MaxMessageSize int
	// IdleTimeout is how long a connection may sit between requests.
	IdleTimeout time.Duration
	// WriteTimeout bounds one reply write, so a client that stops reading cannot pin
	// a goroutine indefinitely.
	WriteTimeout time.Duration
	// Logger receives connection-level events. Nil discards them.
	Logger *slog.Logger
	// Observer receives one Event per served RPC, for metrics. Nil does nothing.
	// It runs on the connection's goroutine before the reply is written, so it must
	// be fast and must not panic.
	Observer Observer
}

// withDefaults fills in the zero values.
func (c Config) withDefaults() Config {
	if c.MaxConnections <= 0 {
		c.MaxConnections = defaultMaxConnections
	}
	if c.MaxInFlight <= 0 {
		c.MaxInFlight = defaultMaxInFlight
	}
	if c.MaxMessageSize <= 0 {
		c.MaxMessageSize = defaultMaxMessageSize
	}
	if c.IdleTimeout <= 0 {
		c.IdleTimeout = defaultIdleTimeout
	}
	if c.WriteTimeout <= 0 {
		c.WriteTimeout = defaultWriteTimeout
	}
	if c.Logger == nil {
		c.Logger = slog.New(slog.DiscardHandler)
	}
	return c
}

// Server serves NFSv3 and MOUNTv3 from one backend. A Server is safe for concurrent use
// and may serve more than one listener.
type Server struct {
	cfg  Config
	disp *dispatcher
}

// New builds a server from a configuration.
func New(cfg Config) (*Server, error) {
	if cfg.Backend == nil {
		return nil, ErrNoBackend
	}
	cfg = cfg.withDefaults()
	nfs := &nfsService{backend: cfg.Backend}
	mount := newMountService(cfg.Backend, cfg.Exports)
	disp := newDispatcher(nfs.program(), mount.program())
	disp.observe = cfg.Observer
	return &Server{cfg: cfg, disp: disp}, nil
}

// Serve accepts connections until ctx is cancelled or the listener fails, then waits for
// the connections it accepted to finish.
//
// Cancelling ctx closes the listener, which is what unblocks the accept loop; a
// cancelled context is not an error, so a clean shutdown returns nil.
func (s *Server) Serve(ctx context.Context, l net.Listener) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var closeOnce sync.Once
	closeListener := func() { closeOnce.Do(func() { _ = l.Close() }) }
	go func() {
		<-ctx.Done()
		closeListener()
	}()
	defer closeListener()

	var (
		conns sync.WaitGroup
		slots = make(chan struct{}, s.cfg.MaxConnections)
	)
	defer conns.Wait()

	for {
		conn, err := l.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return fmt.Errorf("nfsd: accept: %w", err)
		}
		select {
		case slots <- struct{}{}:
		default:
			s.cfg.Logger.Warn("nfsd: connection refused, at capacity",
				"remote", conn.RemoteAddr().String(), "limit", s.cfg.MaxConnections)
			_ = conn.Close()
			continue
		}
		conns.Add(1)
		go func() {
			defer conns.Done()
			defer func() { <-slots }()
			defer conn.Close()
			s.serveConn(ctx, conn)
		}()
	}
}

// Serve runs a server with default limits on one listener, serving one backend. It is
// the short form of [New] followed by [Server.Serve].
func Serve(ctx context.Context, l net.Listener, backend Backend) error {
	s, err := New(Config{Backend: backend})
	if err != nil {
		return err
	}
	return s.Serve(ctx, l)
}

// serveConn reads requests from one connection until it ends.
//
// Replies are written by a single goroutine, because two handlers writing record
// fragments to the same socket would interleave them into nonsense. Requests themselves
// are handled concurrently and may complete out of order, which ONC RPC allows: a reply
// is matched to its call by transaction id, not by arrival.
func (s *Server) serveConn(ctx context.Context, conn net.Conn) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Closing the connection is what unblocks a read that is waiting on a client with
	// nothing to say. Without this a shutdown would wait out the idle timeout on every
	// connection that happened to be quiet, which is minutes.
	go func() {
		<-ctx.Done()
		_ = conn.Close()
	}()

	client := clientAddr(conn.RemoteAddr())
	replies := make(chan []byte, s.cfg.MaxInFlight)

	var writer sync.WaitGroup
	writer.Add(1)
	go func() {
		defer writer.Done()
		defer cancel()
		for reply := range replies {
			if err := conn.SetWriteDeadline(time.Now().Add(s.cfg.WriteTimeout)); err != nil {
				return
			}
			if err := writeRecord(conn, reply); err != nil {
				s.cfg.Logger.Debug("nfsd: write failed", "remote", conn.RemoteAddr().String(), "err", err)
				// Drain rather than return, so handlers still holding a
				// reply are not left blocked on a channel nobody reads.
				_ = conn.Close()
				for range replies {
				}
				return
			}
		}
	}()

	var handlers sync.WaitGroup
	inFlight := make(chan struct{}, s.cfg.MaxInFlight)

	for {
		if err := conn.SetReadDeadline(time.Now().Add(s.cfg.IdleTimeout)); err != nil {
			break
		}
		msg, err := readRecord(conn, s.cfg.MaxMessageSize)
		if err != nil {
			if !errors.Is(err, io.EOF) && ctx.Err() == nil {
				s.cfg.Logger.Debug("nfsd: connection ended",
					"remote", conn.RemoteAddr().String(), "err", err)
			}
			break
		}
		select {
		case inFlight <- struct{}{}:
		case <-ctx.Done():
			goto done
		}
		handlers.Add(1)
		go func() {
			defer handlers.Done()
			defer func() { <-inFlight }()
			reply, ok := s.handle(ctx, msg, client)
			if !ok {
				return
			}
			select {
			case replies <- reply:
			case <-ctx.Done():
			}
		}()
	}

done:
	handlers.Wait()
	close(replies)
	writer.Wait()
}

// handle dispatches one message, returning the reply to write. It reports false when
// the message cannot be answered at all.
//
// The recover is defence in depth rather than a design: every decoder in this package is
// bounds-checked and the fuzzers assert as much. But a panic in one request would
// otherwise take the appliance down with it, and a file server that dies because a
// client sent something odd is worse than one that logs and carries on.
func (s *Server) handle(ctx context.Context, msg []byte, client netip.Addr) (reply []byte, ok bool) {
	defer func() {
		if r := recover(); r != nil {
			s.cfg.Logger.Error("nfsd: panic serving request",
				"client", client.String(), "panic", fmt.Sprint(r), "stack", string(debug.Stack()))
			reply, ok = nil, false
		}
	}()
	enc := NewEncoder(512)
	if err := s.disp.dispatch(ctx, msg, client, enc); err != nil {
		return nil, false
	}
	// The encoder's buffer is reused as the reply is built, so the bytes handed to
	// the writer must be the encoder's own and the encoder must not be touched again.
	return enc.Bytes(), true
}

// clientAddr extracts a comparable address from a connection.
//
// An IPv4-mapped IPv6 address is unmapped, so an allowlist written in terms of 192.0.2.1
// matches a client that arrived over a dual-stack socket as ::ffff:192.0.2.1. A listener
// that is not TCP yields an invalid address, which no sensible allowlist matches.
func clientAddr(a net.Addr) netip.Addr {
	switch t := a.(type) {
	case *net.TCPAddr:
		if ip, ok := netip.AddrFromSlice(t.IP); ok {
			return ip.Unmap()
		}
	}
	if ap, err := netip.ParseAddrPort(a.String()); err == nil {
		return ap.Addr().Unmap()
	}
	return netip.Addr{}
}
