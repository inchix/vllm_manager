// Command modelfsd serves one local directory read-only over NFSv3/MOUNTv3 on TCP.
//
//	modelfsd --export <dir> --listen <ip:port> --allow <cidr,cidr,...> \
//	         [--readahead 8m] [--metrics <ip:port>] [--export-name <path>]
//
// It is the storage role's daemon from docs/03-storage-modelfsd.md: userspace, non-root,
// read-only, one export, no config file and no state. WRITE and every other mutating
// procedure return NFS3ERR_ROFS; a stalled disk returns NFS3ERR_IO rather than hanging the
// client. The wire core and path containment are ported from the sibling project wgshare.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/netip"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/inchix/vllm_manager/storage/modelfsd/internal/localfs"
	"github.com/inchix/vllm_manager/storage/modelfsd/internal/metrics"
	"github.com/inchix/vllm_manager/storage/modelfsd/internal/nfsd"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "modelfsd:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("modelfsd", flag.ContinueOnError)
	var (
		export     = fs.String("export", "", "absolute path of the directory to serve, read-only (required)")
		listen     = fs.String("listen", "", "address to bind, ip:port, e.g. 172.16.254.10:2049 (required)")
		allow      = fs.String("allow", "", "comma-separated client CIDRs permitted to mount and read (required)")
		readahead  = fs.String("readahead", "8m", "read-ahead window per stream, e.g. 8m, 256k, 0 to disable")
		metricsAt  = fs.String("metrics", "", "if set, expose counters over HTTP at this ip:port")
		exportName = fs.String("export-name", "", "mount path clients spell; default is the export dir path")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *export == "" || *listen == "" {
		return errors.New("--export and --listen are required")
	}
	if !filepath.IsAbs(*export) {
		return fmt.Errorf("--export must be an absolute path, got %q", *export)
	}
	if *allow == "" {
		return errors.New("--allow is required (pass one or more client CIDRs)")
	}
	allowNets, err := parseCIDRs(*allow)
	if err != nil {
		return err
	}
	raBytes, err := parseSize(*readahead)
	if err != nil {
		return fmt.Errorf("--readahead: %w", err)
	}

	share := shareName(*export, *exportName)

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	m := metrics.New()

	backend, err := localfs.New(localfs.Config{
		ExportDir: *export,
		Share:     share,
		Allow:     allowNets,
		Readahead: raBytes,
		OnRead:    m.AddBytes,
	})
	if err != nil {
		return fmt.Errorf("open export: %w", err)
	}
	defer backend.Close()

	server, err := nfsd.New(nfsd.Config{
		Backend:  backend,
		Exports:  []string{share},
		Logger:   logger,
		Observer: m.Observe,
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	lc := net.ListenConfig{}
	l, err := lc.Listen(ctx, "tcp", *listen)
	if err != nil {
		return fmt.Errorf("listen %s: %w", *listen, err)
	}

	if *metricsAt != "" {
		startMetrics(ctx, *metricsAt, m, logger)
	}

	logger.Info("modelfsd serving",
		"export", *export, "share", "/"+share, "listen", l.Addr().String(),
		"allow", *allow, "readahead_bytes", raBytes)

	if err := server.Serve(ctx, l); err != nil {
		return err
	}
	logger.Info("modelfsd stopped")
	return nil
}

// startMetrics runs the counters HTTP endpoint until ctx is cancelled.
func startMetrics(ctx context.Context, addr string, m *metrics.Metrics, logger *slog.Logger) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", m)
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()
	go func() {
		logger.Info("modelfsd metrics", "listen", addr, "path", "/metrics")
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("modelfsd metrics server failed", "err", err)
		}
	}()
}

// shareName derives the advertised mount path. An explicit --export-name wins; otherwise
// the export directory's own absolute path is used, with its leading slash stripped, so a
// server for /export/llm_models is mounted as <ip>:/export/llm_models.
func shareName(export, override string) string {
	name := override
	if name == "" {
		name = filepath.Clean(export)
	}
	return strings.Trim(name, "/")
}

// parseCIDRs parses a comma-separated list of CIDR networks.
func parseCIDRs(s string) ([]netip.Prefix, error) {
	var out []netip.Prefix
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		p, err := netip.ParsePrefix(part)
		if err != nil {
			// Accept a bare address as a host route.
			if addr, aerr := netip.ParseAddr(part); aerr == nil {
				out = append(out, netip.PrefixFrom(addr, addr.BitLen()))
				continue
			}
			return nil, fmt.Errorf("bad CIDR %q: %w", part, err)
		}
		out = append(out, p.Masked())
	}
	if len(out) == 0 {
		return nil, errors.New("no valid CIDRs")
	}
	return out, nil
}

// parseSize parses a byte size with an optional k/m/g suffix (powers of 1024).
func parseSize(s string) (int64, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return 0, errors.New("empty size")
	}
	mult := int64(1)
	switch s[len(s)-1] {
	case 'k':
		mult, s = 1<<10, s[:len(s)-1]
	case 'm':
		mult, s = 1<<20, s[:len(s)-1]
	case 'g':
		mult, s = 1<<30, s[:len(s)-1]
	}
	n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	if n < 0 {
		return 0, errors.New("size must not be negative")
	}
	return n * mult, nil
}
