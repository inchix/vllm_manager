package localfs

import (
	"context"
	"errors"
	"io"
	"sync"
	"time"
)

// Read-ahead over local disk, adapted from wgshare's internal/lanserve/readahead.go.
//
// In wgshare each chunk was a tunnel round trip to a remote peer, and read-ahead existed
// to overlap those round trips instead of queueing them. modelfsd's byte source is a
// local file, so the same machinery serves a smaller but real purpose: it fetches whole
// chunks (larger than the client's individual READ), keeps a bounded window of them warm,
// coalesces concurrent readers of the same chunk into one physical read, and — the
// property the storage design actually cares about — bounds every physical read with a
// timeout, so a stalled or vanished disk yields an error promptly rather than a client
// wedged in uninterruptible sleep. See docs/03-storage-modelfsd.md.
//
// What was dropped from the wgshare original: the peer transport, the Fetcher/Resolver
// indirection and the remote-share bookkeeping. The source here is a plain positional
// read against an *os.File the backend already holds open.

const (
	// defaultChunkSize is the granularity of the read cache. A client READ is at most
	// rtmax (512 KiB); a chunk is smaller than that so a single READ still maps onto a
	// bounded, whole number of chunks while giving read-ahead something to prefetch.
	defaultChunkSize = 256 << 10

	// defaultReadTimeout bounds one physical read. A read that has taken this long is
	// treated as a failed device rather than waited on indefinitely; the backend maps
	// the resulting error onto NFS3ERR_IO.
	defaultReadTimeout = 20 * time.Second

	// maxPrefetchWorkers bounds background prefetches across every reader. A prefetch
	// that cannot start is one nobody is waiting for, so it is simply skipped.
	maxPrefetchWorkers = 8

	// minCachedChunks is the floor on the warm window, so read-ahead always has room to
	// hold what it fetched even when the configured window is tiny.
	minCachedChunks = 8
)

// errReadTimeout is returned when a physical read outruns the read timeout. It is a
// distinct sentinel so the backend can recognise it, but any non-EOF error from a read
// already becomes NFS3ERR_IO, which is the honest answer here.
var errReadTimeout = errors.New("localfs: read timed out")

// source reads bytes for a file id at an absolute offset. It blocks and takes no context,
// exactly like io.ReaderAt.ReadAt; the timeout is applied around it by the cache.
type source func(fileID uint64, p []byte, off int64) (int, error)

// chunkKey identifies one chunk of one file.
type chunkKey struct {
	fileID uint64
	chunk  int64
}

// pending is one fetch somebody is already doing. A hand-rolled singleflight: one key,
// one result, no cancellation of the shared work by whoever merely waited on it.
type pending struct {
	done chan struct{}
	data []byte
	err  error
}

// readCache is the read-ahead chunk cache.
type readCache struct {
	src         source
	chunkSize   int64
	ahead       int // chunks to prefetch ahead of a sequential reader
	readTimeout time.Duration
	maxChunks   int

	mu        sync.Mutex
	chunks    map[chunkKey][]byte
	order     []chunkKey // insertion order, for LRU-ish eviction
	inflight  map[chunkKey]*pending
	lastChunk map[uint64]int64 // last chunk each reader touched, for sequentiality

	prefetch chan struct{}
}

// newReadCache builds a cache. readahead is the window in bytes to prefetch ahead of a
// sequential reader (rounded to whole chunks); readTimeout bounds one physical read.
func newReadCache(src source, readahead int64, readTimeout time.Duration) *readCache {
	if readTimeout <= 0 {
		readTimeout = defaultReadTimeout
	}
	chunkSize := int64(defaultChunkSize)
	ahead := 0
	if readahead > 0 {
		ahead = int((readahead + chunkSize - 1) / chunkSize)
	}
	maxChunks := minCachedChunks
	if n := ahead * (maxPrefetchWorkers + 1); n > maxChunks {
		maxChunks = n
	}
	return &readCache{
		src:         src,
		chunkSize:   chunkSize,
		ahead:       ahead,
		readTimeout: readTimeout,
		maxChunks:   maxChunks,
		chunks:      make(map[chunkKey][]byte),
		inflight:    make(map[chunkKey]*pending),
		lastChunk:   make(map[uint64]int64),
		prefetch:    make(chan struct{}, maxPrefetchWorkers),
	}
}

// ReadAt fills p from off, following io.ReaderAt: a short read at end-of-file is reported
// with io.EOF. It serves from the chunk cache, fetching whole chunks as needed, and
// triggers background read-ahead for a reader that is advancing.
func (c *readCache) ReadAt(ctx context.Context, fileID uint64, p []byte, off int64) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	total := 0
	for total < len(p) {
		pos := off + int64(total)
		idx := pos / c.chunkSize
		within := pos - idx*c.chunkSize

		data, err := c.chunk(ctx, fileID, idx)
		if err != nil {
			return total, err
		}
		c.readAhead(fileID, idx)

		if within >= int64(len(data)) {
			// The offset is past everything this chunk holds: end of file.
			return total, io.EOF
		}
		n := copy(p[total:], data[within:])
		total += n

		if int64(len(data)) < c.chunkSize {
			// A short chunk is the last one. If the caller still wanted more, that
			// more does not exist.
			if total < len(p) {
				return total, io.EOF
			}
			return total, nil
		}
	}
	return total, nil
}

// chunk returns one chunk, from the cache, by joining an in-flight fetch, or by fetching
// it. Exactly one caller performs the physical read; the rest wait on it.
func (c *readCache) chunk(ctx context.Context, fileID uint64, idx int64) ([]byte, error) {
	key := chunkKey{fileID: fileID, chunk: idx}

	c.mu.Lock()
	if d, ok := c.chunks[key]; ok {
		c.mu.Unlock()
		return d, nil
	}
	if p, ok := c.inflight[key]; ok {
		c.mu.Unlock()
		return waitFor(ctx, p)
	}
	p := &pending{done: make(chan struct{})}
	c.inflight[key] = p
	c.mu.Unlock()

	data, err := c.fetch(ctx, fileID, idx)

	c.mu.Lock()
	delete(c.inflight, key)
	if err == nil {
		c.storeLocked(key, data)
	}
	c.mu.Unlock()

	p.data, p.err = data, err
	close(p.done)
	return data, err
}

// waitFor blocks for a fetch somebody else started. The caller's context still applies:
// a client that goes away is not held by another client's slow read.
func waitFor(ctx context.Context, p *pending) ([]byte, error) {
	select {
	case <-p.done:
		return p.data, p.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// fetch performs one physical, timeout-bounded read of a whole chunk. An EOF from the
// source is not an error: it just means the final chunk is short.
func (c *readCache) fetch(ctx context.Context, fileID uint64, idx int64) ([]byte, error) {
	buf := make([]byte, c.chunkSize)
	n, err := c.physRead(ctx, fileID, buf, idx*c.chunkSize)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	if n < 0 {
		n = 0
	}
	return buf[:n], nil
}

// physRead runs one source read under a timeout and the caller's context. On timeout or
// cancellation the underlying read is abandoned — its goroutine finishes on its own when
// the read finally returns — and an error is reported at once rather than waited out.
func (c *readCache) physRead(ctx context.Context, fileID uint64, buf []byte, off int64) (int, error) {
	type result struct {
		n   int
		err error
	}
	ch := make(chan result, 1)
	go func() {
		n, err := c.src(fileID, buf, off)
		ch <- result{n, err}
	}()

	timer := time.NewTimer(c.readTimeout)
	defer timer.Stop()
	select {
	case r := <-ch:
		return r.n, r.err
	case <-ctx.Done():
		return 0, ctx.Err()
	case <-timer.C:
		return 0, errReadTimeout
	}
}

// readAhead starts background fetches for the chunks after idx, but only for a reader that
// is advancing — a sequential load, which is exactly the model-weight access pattern, and
// not a seek-heavy one where prefetch would spend bandwidth nobody reaches.
func (c *readCache) readAhead(fileID uint64, idx int64) {
	if c.ahead <= 0 {
		return
	}
	c.mu.Lock()
	advancing := c.lastChunk[fileID] == idx-1 || c.lastChunk[fileID] == idx
	c.lastChunk[fileID] = idx
	if len(c.lastChunk) > c.maxChunks {
		for k := range c.lastChunk {
			delete(c.lastChunk, k)
			break
		}
	}
	c.mu.Unlock()
	if !advancing {
		return
	}

	for n := int64(1); n <= int64(c.ahead); n++ {
		next := chunkKey{fileID: fileID, chunk: idx + n}
		c.mu.Lock()
		_, cached := c.chunks[next]
		_, busy := c.inflight[next]
		c.mu.Unlock()
		if cached || busy {
			continue
		}
		select {
		case c.prefetch <- struct{}{}:
		default:
			return // every worker is busy; the reader will ask for it in a moment
		}
		go func(chunk int64) {
			defer func() { <-c.prefetch }()
			ctx, cancel := context.WithTimeout(context.Background(), c.readTimeout)
			defer cancel()
			// Errors are dropped: nobody is waiting, and the read that eventually
			// wants the chunk fails on its own terms with its own error.
			_, _ = c.chunk(ctx, fileID, chunk)
		}(idx + n)
	}
}

// storeLocked adds a chunk under the lock, evicting the oldest once the window is full.
func (c *readCache) storeLocked(key chunkKey, data []byte) {
	if _, ok := c.chunks[key]; ok {
		return
	}
	c.chunks[key] = data
	c.order = append(c.order, key)
	for len(c.order) > c.maxChunks {
		old := c.order[0]
		c.order = c.order[1:]
		delete(c.chunks, old)
	}
}
