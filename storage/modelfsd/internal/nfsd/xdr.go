package nfsd

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// XDR (RFC 4506) encoding and decoding.
//
// Written here rather than taken from a library because it is a small, closed format
// and this is the first thing a hostile LAN client reaches. A decoder we own is one we
// can hold to a single rule: every read is bounds-checked against what is actually in
// the buffer before anything is allocated, so a declared length of four billion costs
// an error rather than four gigabytes.

// Decoding errors. They are sentinels because the RPC layer maps them onto
// GARBAGE_ARGS, which is the only reply a client can learn anything from.
var (
	// ErrTruncated means the input ended in the middle of a value.
	ErrTruncated = errors.New("nfsd: xdr: input truncated")
	// ErrTooLong means a length prefix exceeded the limit the field allows.
	ErrTooLong = errors.New("nfsd: xdr: field exceeds its limit")
	// ErrBadBool means a boolean was encoded as something other than zero or one.
	ErrBadBool = errors.New("nfsd: xdr: boolean is neither zero nor one")
)

// Decoder reads XDR-encoded values from a fixed buffer.
//
// It is not safe for concurrent use, and it never retains a reference to the buffer in
// anything it returns: opaque fields are copied, because a filehandle handed to a
// backend outlives the connection buffer it arrived in.
type Decoder struct {
	buf []byte
	off int
}

// NewDecoder prepares to decode buf. The buffer is not copied and must not be modified
// while the decoder is in use.
func NewDecoder(buf []byte) *Decoder { return &Decoder{buf: buf} }

// Remaining reports how many bytes are left undecoded.
func (d *Decoder) Remaining() int { return len(d.buf) - d.off }

// take advances over n bytes and returns them, without accounting for XDR's padding to
// a four-byte boundary. Every other read goes through here, so this is the single place
// a length is checked against reality.
func (d *Decoder) take(n int) ([]byte, error) {
	if n < 0 || n > d.Remaining() {
		return nil, ErrTruncated
	}
	b := d.buf[d.off : d.off+n]
	d.off += n
	return b, nil
}

// pad4 rounds a length up to XDR's four-byte alignment.
func pad4(n int) int { return (n + 3) &^ 3 }

// Uint32 decodes an unsigned 32-bit integer.
func (d *Decoder) Uint32() (uint32, error) {
	b, err := d.take(4)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint32(b), nil
}

// Int32 decodes a signed 32-bit integer.
func (d *Decoder) Int32() (int32, error) {
	v, err := d.Uint32()
	return int32(v), err
}

// Uint64 decodes an unsigned 64-bit integer.
func (d *Decoder) Uint64() (uint64, error) {
	b, err := d.take(8)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint64(b), nil
}

// Int64 decodes a signed 64-bit integer.
func (d *Decoder) Int64() (int64, error) {
	v, err := d.Uint64()
	return int64(v), err
}

// Bool decodes a boolean.
//
// A value other than zero or one is rejected rather than treated as true. XDR defines
// only those two encodings, and a decoder that quietly accepts more gives two peers two
// different readings of the same bytes.
func (d *Decoder) Bool() (bool, error) {
	v, err := d.Uint32()
	if err != nil {
		return false, err
	}
	switch v {
	case 0:
		return false, nil
	case 1:
		return true, nil
	}
	return false, fmt.Errorf("%w: %d", ErrBadBool, v)
}

// Fixed decodes fixed-length opaque data of exactly n bytes, plus its padding. The
// result is a copy.
func (d *Decoder) Fixed(n int) ([]byte, error) {
	if n < 0 {
		return nil, ErrTruncated
	}
	b, err := d.take(pad4(n))
	if err != nil {
		return nil, err
	}
	out := make([]byte, n)
	copy(out, b[:n])
	return out, nil
}

// length decodes a length prefix and checks it against both the caller's limit and the
// bytes actually present. Checking against the buffer as well as the limit is what
// makes a declared length harmless: a claim can never be larger than the message that
// carried it.
func (d *Decoder) length(max int) (int, error) {
	v, err := d.Uint32()
	if err != nil {
		return 0, err
	}
	if max < 0 || uint64(v) > uint64(max) {
		return 0, fmt.Errorf("%w: %d bytes, limit is %d", ErrTooLong, v, max)
	}
	if uint64(v) > uint64(d.Remaining()) {
		return 0, ErrTruncated
	}
	return int(v), nil
}

// Opaque decodes variable-length opaque data of at most max bytes. The result is a
// copy, and is never nil for a zero-length field.
func (d *Decoder) Opaque(max int) ([]byte, error) {
	n, err := d.length(max)
	if err != nil {
		return nil, err
	}
	b, err := d.take(pad4(n))
	if err != nil {
		return nil, err
	}
	out := make([]byte, n)
	copy(out, b[:n])
	return out, nil
}

// String decodes a variable-length string of at most max bytes.
//
// The bytes are not validated as UTF-8. Linux permits filenames that are not valid
// UTF-8, and mangling them here would make those files unreachable; a caller rendering
// a name for a browser sanitises it there.
func (d *Decoder) String(max int) (string, error) {
	b, err := d.Opaque(max)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Count decodes an array length of at most max elements.
func (d *Decoder) Count(max int) (int, error) {
	v, err := d.Uint32()
	if err != nil {
		return 0, err
	}
	if max < 0 || uint64(v) > uint64(max) {
		return 0, fmt.Errorf("%w: %d elements, limit is %d", ErrTooLong, v, max)
	}
	// Every XDR element occupies at least four bytes, so a count that cannot fit in
	// the remaining input is a lie and is refused before anything is allocated.
	if uint64(v)*4 > uint64(d.Remaining()) {
		return 0, ErrTruncated
	}
	return int(v), nil
}

// Skip advances over n bytes of padded data.
func (d *Decoder) Skip(n int) error {
	_, err := d.take(pad4(n))
	return err
}

// DecodeArray decodes a counted array of at most max elements using decode for each.
//
// The result is grown by appending rather than preallocated from the count: the count
// has already been checked against the input, but growing on demand means even a
// pathological limit costs only what the input can actually justify.
func DecodeArray[T any](d *Decoder, max int, decode func(*Decoder) (T, error)) ([]T, error) {
	n, err := d.Count(max)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	out := make([]T, 0, min(n, 64))
	for i := 0; i < n; i++ {
		v, err := decode(d)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// Encoder builds an XDR-encoded message.
//
// Encoding cannot fail: every value the server emits is one it produced itself, and a
// length it cannot represent is a bug here rather than an input to validate. The one
// exception is guarded at the point of use, in the record writer.
type Encoder struct {
	buf []byte
}

// NewEncoder returns an encoder with room for n bytes reserved up front.
func NewEncoder(n int) *Encoder { return &Encoder{buf: make([]byte, 0, n)} }

// Bytes returns the encoded message. The slice aliases the encoder's buffer and is
// invalidated by any further write.
func (e *Encoder) Bytes() []byte { return e.buf }

// Len reports how many bytes have been encoded.
func (e *Encoder) Len() int { return len(e.buf) }

// Reset discards the encoded message, keeping the buffer for reuse.
func (e *Encoder) Reset() { e.buf = e.buf[:0] }

// Truncate discards everything encoded after the first n bytes. It exists so a reply
// can be abandoned part-way and rebuilt as an error reply.
func (e *Encoder) Truncate(n int) {
	if n >= 0 && n <= len(e.buf) {
		e.buf = e.buf[:n]
	}
}

// Uint32 encodes an unsigned 32-bit integer.
func (e *Encoder) Uint32(v uint32) {
	e.buf = binary.BigEndian.AppendUint32(e.buf, v)
}

// Int32 encodes a signed 32-bit integer.
func (e *Encoder) Int32(v int32) { e.Uint32(uint32(v)) }

// Uint64 encodes an unsigned 64-bit integer.
func (e *Encoder) Uint64(v uint64) {
	e.buf = binary.BigEndian.AppendUint64(e.buf, v)
}

// Int64 encodes a signed 64-bit integer.
func (e *Encoder) Int64(v int64) { e.Uint64(uint64(v)) }

// Bool encodes a boolean.
func (e *Encoder) Bool(v bool) {
	if v {
		e.Uint32(1)
		return
	}
	e.Uint32(0)
}

// Fixed encodes fixed-length opaque data, padded to a four-byte boundary. No length
// prefix is written: the length is part of the format both ends agreed on.
func (e *Encoder) Fixed(b []byte) {
	e.buf = append(e.buf, b...)
	e.pad(len(b))
}

// Opaque encodes variable-length opaque data with its length prefix.
func (e *Encoder) Opaque(b []byte) {
	e.Uint32(uint32(len(b)))
	e.Fixed(b)
}

// String encodes a variable-length string with its length prefix.
func (e *Encoder) String(s string) {
	e.Uint32(uint32(len(s)))
	e.buf = append(e.buf, s...)
	e.pad(len(s))
}

// pad writes XDR's zero padding out to a four-byte boundary.
func (e *Encoder) pad(n int) {
	for i := n; i < pad4(n); i++ {
		e.buf = append(e.buf, 0)
	}
}

// EncodeArray encodes a counted array using encode for each element.
func EncodeArray[T any](e *Encoder, xs []T, encode func(*Encoder, T)) {
	e.Uint32(uint32(len(xs)))
	for _, x := range xs {
		encode(e, x)
	}
}

// opaqueSize reports the encoded size of variable-length opaque data of n bytes,
// including its length prefix and padding. Used by READDIR's size accounting, which has
// to know what a reply will cost before it commits to building it.
func opaqueSize(n int) int { return 4 + pad4(n) }

// clampTime folds a Unix time into the unsigned seconds field NFSv3 uses. Times outside
// the representable range are clamped rather than wrapped: a wrapped mtime makes a
// client's cache behave erratically, whereas a clamped one is merely wrong in a way it
// can see.
func clampTime(sec int64) uint32 {
	if sec < 0 {
		return 0
	}
	if sec > math.MaxUint32 {
		return math.MaxUint32
	}
	return uint32(sec)
}
