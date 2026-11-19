#!/usr/bin/env python3
"""Expose a Hugging Face Xet download as a local HTTP Range source.

The beellama.cpp `--hf-stream-vram` engine can consume this URL and push bytes
into VRAM while the Xet download is still running.  The payload is written
sequentially to a sparse local file; range requests block until the requested
byte is available.

This helper is deliberately standalone: it only needs `huggingface_hub` and
`hf_xet`, both already present in the Colab image used by collabosm.
"""
import argparse
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import hf_xet
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils._xet import refresh_xet_connection_info


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def parse_hf_url(url):
    """Return (repo, revision, filename) from a huggingface.co resolve URL."""
    p = urlparse(url)
    if p.netloc not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError("not a huggingface.co URL: %s" % url)
    parts = [x for x in p.path.split("/") if x]
    try:
        i = parts.index("resolve")
    except ValueError:
        raise ValueError("URL has no /resolve/ segment: %s" % url)
    repo = "/".join(parts[:i])
    revision = parts[i + 1]
    filename = "/".join(parts[i + 2:])
    if not repo or not revision or not filename:
        raise ValueError("incomplete HF resolve URL: %s" % url)
    return repo, revision, filename


class Source:
    def __init__(self, repo, filename, revision, local_path, token=None,
                 disk=True, window_mb=1024):
        self.repo = repo
        self.filename = filename
        self.revision = revision
        self.local_path = local_path
        self.token = token
        self.disk = disk
        self.window_limit = max(64, window_mb) * 1024 * 1024
        self.window = deque()
        self.window_bytes = 0
        self.size = 0
        self.etag = ""
        self.available = 0
        self.done = False
        self.error = None
        self.cond = threading.Condition()
        self.fd = -1
        self.thread = None

    def start(self):
        url = hf_hub_url(self.repo, self.filename, revision=self.revision)
        headers = {}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        meta = get_hf_file_metadata(url, headers=headers or None)
        if meta.xet_file_data is None:
            raise RuntimeError("HF returned no Xet file data for %s" % url)
        self.size = int(meta.size)
        self.etag = meta.etag or ""
        conn = refresh_xet_connection_info(file_data=meta.xet_file_data, headers=headers)
        if self.disk:
            os.makedirs(os.path.dirname(os.path.abspath(self.local_path)), exist_ok=True)
            self.fd = os.open(self.local_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(self.fd, self.size)
        self.thread = threading.Thread(target=self._run, args=(meta, conn, headers), daemon=True)
        self.thread.start()

    def _run(self, meta, conn, headers):
        try:
            session = hf_xet.XetSession()
            group = session.new_download_stream_group(
                endpoint=conn.endpoint,
                token=conn.access_token,
                token_expiry_unix_secs=conn.expiration_unix_epoch,
                token_refresh_url=meta.xet_file_data.refresh_route,
                token_refresh_headers=headers,
            )
            info = hf_xet.XetFileInfo(meta.xet_file_data.file_hash, self.size)
            off = 0
            t0 = time.time()
            last = t0
            for chunk in group.download_stream(info):
                n = len(chunk)
                if self.disk:
                    done = 0
                    while done < n:
                        done += os.pwrite(self.fd, chunk[done:], off + done)
                with self.cond:
                    self.window.append((off, chunk))
                    self.window_bytes += n
                    while self.window_bytes > self.window_limit and len(self.window) > 1:
                        _, old = self.window.popleft()
                        self.window_bytes -= len(old)
                    self.available = off + n
                    self.cond.notify_all()
                off += n
                if time.time() - last > 2.0:
                    log("[xet] %.1f/%.1f MB  %.1f MB/s" % (
                        off / 1e6, self.size / 1e6, off / (time.time() - t0) / 1e6))
                    last = time.time()
            with self.cond:
                self.done = True
                self.available = off
                self.cond.notify_all()
            log("[xet] done %.1f MB in %.1fs (%.1f MB/s)" % (
                off / 1e6, time.time() - t0, off / (time.time() - t0) / 1e6))
        except Exception as e:
            with self.cond:
                self.error = e
                self.cond.notify_all()
            log("[xet] ERROR: %r" % (e,))

    def wait_for(self, end):
        with self.cond:
            while self.available < end and not self.done and self.error is None:
                self.cond.wait(timeout=1.0)
            if self.error is not None:
                raise RuntimeError(str(self.error))
            if self.available < end:
                raise RuntimeError("download ended at %d before %d" % (self.available, end))

    def _read_window(self, start, end):
        out = bytearray()
        with self.cond:
            for off, chunk in self.window:
                a = max(start, off)
                b = min(end, off + len(chunk))
                if a < b:
                    out += chunk[a - off:b - off]
        if len(out) == end - start:
            return bytes(out)
        return None

    def read(self, start, end):
        data = self._read_window(start, end)
        if data is not None:
            return data
        if not self.disk:
            raise RuntimeError("range [%d,%d) left the memory window" % (start, end))
        out = bytearray()
        pos = start
        while pos < end:
            n = min(1 << 20, end - pos)
            b = os.pread(self.fd, n, pos)
            if not b:
                raise RuntimeError("short read at %d" % pos)
            out += b
            pos += len(b)
        return bytes(out)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    source = None
    verbose = False

    def log_message(self, fmt, *args):
        if self.verbose:
            log("[http] " + (fmt % args))

    def _send_common(self):
        self.send_header("Accept-Ranges", "bytes")
        if self.source.etag:
            self.send_header("ETag", self.source.etag)

    def do_HEAD(self):
        src = self.source
        self.send_response(200)
        self.send_header("Content-Length", str(src.size))
        self._send_common()
        self.end_headers()

    def do_GET(self):
        src = self.source
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(200)
            self.send_header("Content-Length", str(src.size))
            self._send_common()
            self.end_headers()
            try:
                for off in range(0, src.size, 1 << 20):
                    self.wfile.write(src.read(off, min(off + (1 << 20), src.size)))
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        try:
            units, _, spec = rng.partition("=")
            if units.strip().lower() != "bytes":
                raise ValueError("bad range")
            a, _, b = spec.partition("-")
            begin = int(a) if a else 0
            last = int(b) if b else src.size - 1
            last = min(last, src.size - 1)
            if begin > last:
                raise ValueError("bad range")
            src.wait_for(last + 1)
            data = src.read(begin, last + 1)
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (begin, last, src.size))
            self.send_header("Content-Length", str(len(data)))
            self._send_common()
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log("[http] range error: %r" % (e,))
            try:
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()
            except Exception:
                pass


def stdin_watch():
    # The parent keeps the write end open.  EOF means the server is gone.
    try:
        sys.stdin.buffer.read(1)
    except Exception:
        pass
    os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--local-path", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--token", default="")
    ap.add_argument("--window-mb", type=int, default=1024,
                    help="memory window kept for range reads (MiB)")
    ap.add_argument("--no-disk", action="store_true",
                    help="keep the payload only in the memory window")
    ap.add_argument("--no-stdin-watch", action="store_true",
                    help="do not exit when stdin reaches EOF (shell-started proxy)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.url and not args.repo:
        args.repo, args.revision, args.file = parse_hf_url(args.url)
    if not args.repo or not args.file:
        ap.error("need --url or --repo/--file")

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    local = args.local_path or os.path.join("/tmp", "xet-%d.gguf" % os.getpid())
    src = Source(args.repo, args.file, args.revision, local, token,
                 disk=not args.no_disk, window_mb=args.window_mb)
    src.start()
    Handler.source = src
    Handler.verbose = args.verbose
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.daemon_threads = True
    httpd.request_queue_size = 64
    port = httpd.server_address[1]

    # The C++ parent keeps stdin open; the thread exits the proxy when it closes.
    # A shell-started proxy passes --no-stdin-watch because its stdin is often /dev/null.
    if not args.no_stdin_watch:
        threading.Thread(target=stdin_watch, daemon=True).start()

    print("PORT %d" % port, flush=True)
    print("PATH %s" % local, flush=True)
    print("SIZE %d" % src.size, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
