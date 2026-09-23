// Package metrics exposes a handful of counters for modelfsd over HTTP.
//
// It reuses wgshare's observe pattern: the nfsd dispatcher calls an Observer once per
// served RPC, and that is the single place per-procedure counts and RPC-level errors are
// tallied. Bytes served are counted by the backend, which is the only place that knows a
// READ's size. The exposition is a tiny Prometheus-style text handler; there is no
// external metrics dependency.
package metrics

import (
	"fmt"
	"net/http"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/inchix/vllm_manager/storage/modelfsd/internal/nfsd"
)

// Metrics holds modelfsd's counters. The zero value is not usable; call New.
type Metrics struct {
	started time.Time

	ops    sync.Map // proc name -> *atomic.Int64
	errors atomic.Int64
	bytes  atomic.Int64
}

// New returns a ready Metrics.
func New() *Metrics {
	return &Metrics{started: time.Now()}
}

// Observe records one served RPC. It is the nfsd.Observer for the server, and runs on the
// connection goroutine before the reply is written, so it does only arithmetic.
func (m *Metrics) Observe(e nfsd.Event) {
	name := e.Name
	if name == "" {
		name = "UNKNOWN"
	}
	v, ok := m.ops.Load(name)
	if !ok {
		v, _ = m.ops.LoadOrStore(name, new(atomic.Int64))
	}
	v.(*atomic.Int64).Add(1)
	if e.Status != nfsd.StatusOK {
		m.errors.Add(1)
	}
}

// AddBytes records bytes returned by a READ.
func (m *Metrics) AddBytes(n int) {
	if n > 0 {
		m.bytes.Add(int64(n))
	}
}

// ServeHTTP writes the counters in Prometheus text format.
func (m *Metrics) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	type row struct {
		name  string
		count int64
	}
	var rows []row
	var opsTotal int64
	m.ops.Range(func(k, v any) bool {
		c := v.(*atomic.Int64).Load()
		opsTotal += c
		rows = append(rows, row{name: k.(string), count: c})
		return true
	})
	sort.Slice(rows, func(i, j int) bool { return rows[i].name < rows[j].name })

	fmt.Fprintf(w, "# HELP modelfsd_uptime_seconds Seconds since the server started.\n")
	fmt.Fprintf(w, "# TYPE modelfsd_uptime_seconds gauge\n")
	fmt.Fprintf(w, "modelfsd_uptime_seconds %d\n", int64(time.Since(m.started).Seconds()))

	fmt.Fprintf(w, "# HELP modelfsd_rpc_ops_total RPCs served, by procedure.\n")
	fmt.Fprintf(w, "# TYPE modelfsd_rpc_ops_total counter\n")
	fmt.Fprintf(w, "modelfsd_rpc_ops_total %d\n", opsTotal)
	for _, r := range rows {
		fmt.Fprintf(w, "modelfsd_rpc_ops_by_proc_total{proc=%q} %d\n", r.name, r.count)
	}

	fmt.Fprintf(w, "# HELP modelfsd_rpc_errors_total RPC-level failures (garbage/system/auth).\n")
	fmt.Fprintf(w, "# TYPE modelfsd_rpc_errors_total counter\n")
	fmt.Fprintf(w, "modelfsd_rpc_errors_total %d\n", m.errors.Load())

	fmt.Fprintf(w, "# HELP modelfsd_read_bytes_total Bytes returned by READ.\n")
	fmt.Fprintf(w, "# TYPE modelfsd_read_bytes_total counter\n")
	fmt.Fprintf(w, "modelfsd_read_bytes_total %d\n", m.bytes.Load())
}
