"""A throwaway private CA and TLS endpoints for the catalog client's tests.

The certificates are generated per test session with the ``openssl`` CLI --
nothing secret is committed -- and live for two days. The server
certificate names ``IP:127.0.0.1`` and ``DNS:localhost`` only, so a client
that verifies the peer accepts it for those names and nothing else.

Two endpoints are built on it:

* :class:`FakeClickHouse` -- a tiny HTTP(S) server that records every request
  (method, path, headers, body) and answers through a caller's function, so
  a test can see exactly what reached the wire;
* :class:`TlsTerminator` -- TLS in front of a plain TCP port (the local
  ClickHouse HTTP interface), for the live round trip over verified TLS.
"""

from __future__ import annotations

import shutil
import socket
import ssl
import subprocess
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

_CA_CONFIG = """\
[req]
distinguished_name = dn
prompt = no
x509_extensions = v3_ca
[dn]
CN = DMI catalog test CA
[v3_ca]
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
"""

_LEAF_CONFIG = """\
[req]
distinguished_name = dn
prompt = no
[dn]
CN = {cn}
"""

_LEAF_EXTENSIONS = """\
basicConstraints = CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
subjectAltName = {san}
"""


@dataclass(frozen=True)
class PrivateCa:
    ca_file: Path    # the CA certificate, PEM
    ca_path: Path    # a hashed directory holding only that CA (CURLOPT_CAPATH)
    cert: Path       # server certificate for 127.0.0.1 / localhost
    key: Path
    # Signed by the same CA for a name that is NOT this host: a client that
    # checks the name refuses it even though it trusts the issuer.
    wrong_name_cert: Path
    wrong_name_key: Path

    def server_context(self, *, wrong_name: bool = False) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        if wrong_name:
            context.load_cert_chain(str(self.wrong_name_cert),
                                    str(self.wrong_name_key))
        else:
            context.load_cert_chain(str(self.cert), str(self.key))
        return context


def _openssl(*args: str, cwd: Path) -> str:
    return subprocess.run(["openssl", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def make_private_ca(directory: Path) -> PrivateCa:
    """Generate a CA and a server certificate it signs, under `directory`."""
    if shutil.which("openssl") is None:
        raise RuntimeError("the openssl CLI is required to generate the test CA")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ca.cnf").write_text(_CA_CONFIG)
    _openssl("req", "-x509", "-config", "ca.cnf", "-newkey", "rsa:2048",
             "-nodes", "-keyout", "ca.key", "-out", "ca.pem", "-days", "2",
             cwd=directory)
    for stem, cn, san in (
            ("server", "127.0.0.1", "IP:127.0.0.1,DNS:localhost"),
            ("wrong-name", "not-this-host.invalid",
             "DNS:not-this-host.invalid")):
        (directory / f"{stem}.cnf").write_text(_LEAF_CONFIG.format(cn=cn))
        (directory / f"{stem}.ext").write_text(
            _LEAF_EXTENSIONS.format(san=san))
        _openssl("req", "-new", "-config", f"{stem}.cnf", "-newkey",
                 "rsa:2048", "-nodes", "-keyout", f"{stem}.key", "-out",
                 f"{stem}.csr", cwd=directory)
        _openssl("x509", "-req", "-in", f"{stem}.csr", "-CA", "ca.pem",
                 "-CAkey", "ca.key", "-CAcreateserial", "-out", f"{stem}.pem",
                 "-days", "2", "-extfile", f"{stem}.ext", cwd=directory)
    # CURLOPT_CAPATH reads OpenSSL's hashed layout: <subject hash>.0.
    subject_hash = _openssl("x509", "-hash", "-noout", "-in", "ca.pem",
                            cwd=directory).strip()
    hashed = directory / "hashed"
    hashed.mkdir(exist_ok=True)
    shutil.copy(directory / "ca.pem", hashed / f"{subject_hash}.0")
    return PrivateCa(ca_file=directory / "ca.pem", ca_path=hashed,
                     cert=directory / "server.pem",
                     key=directory / "server.key",
                     wrong_name_cert=directory / "wrong-name.pem",
                     wrong_name_key=directory / "wrong-name.key")


@dataclass
class Request:
    method: str
    path: str
    headers: dict[str, str]   # lower-cased names
    body: bytes


# A responder returns (status, body), or None to close the connection
# without answering (the client sees an empty reply: a transport error).
Responder = Callable[[Request], Optional[tuple[int, bytes]]]


class FakeClickHouse:
    """Records every request and answers through `respond`.

    With `tls`, the listening socket speaks TLS with the private CA's server
    certificate; `handshakes` counts accepted connections and
    `failed_handshakes` those whose handshake failed (a client that refused
    the certificate).
    """

    def __init__(self, respond: Optional[Responder] = None, *,
                 tls: Optional[PrivateCa] = None, wrong_name: bool = False):
        self.requests: list[Request] = []
        self.handshakes = 0
        self.failed_handshakes = 0
        self._respond = respond or (lambda request: (200, b""))
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server's naming
                length = int(self.headers.get("Content-Length", "0"))
                request = Request(
                    method="POST", path=self.path,
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=self.rfile.read(length))
                fake.requests.append(request)
                answer = fake._respond(request)
                if answer is None:
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                status, body = answer
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST  # noqa: N815

            def log_message(self, *args):
                pass

        class _Server(ThreadingHTTPServer):
            daemon_threads = True

            def get_request(self):
                sock, address = self.socket.accept()
                if tls is None:
                    return sock, address
                fake.handshakes += 1
                try:
                    return context.wrap_socket(sock, server_side=True), address
                except (ssl.SSLError, OSError):
                    fake.failed_handshakes += 1
                    sock.close()
                    raise

            def handle_error(self, request, client_address):
                pass

        context = (tls.server_context(wrong_name=wrong_name)
                   if tls is not None else None)
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TlsTerminator:
    """TLS on a local port, forwarded as plain TCP to `target`."""

    def __init__(self, ca: PrivateCa, target: tuple[str, int]):
        self._context = ca.server_context()
        self._target = target
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self.handshakes = 0
        self.failed_handshakes = 0
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(client,),
                             daemon=True).start()

    def _serve(self, client):
        self.handshakes += 1
        try:
            tls = self._context.wrap_socket(client, server_side=True)
        except (ssl.SSLError, OSError):
            self.failed_handshakes += 1
            client.close()
            return
        try:
            upstream = socket.create_connection(self._target)
        except OSError:
            tls.close()
            return
        threading.Thread(target=self._pump, args=(upstream, tls),
                         daemon=True).start()
        self._pump(tls, upstream)

    @staticmethod
    def _pump(source, sink):
        try:
            while data := source.recv(65536):
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for end in (source, sink):
                try:
                    end.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self):
        self._listener.close()
