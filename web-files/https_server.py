import http.server
import ssl

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

httpd = http.server.HTTPServer(('0.0.0.0', PORT), Handler)

# Create SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile="certificate.crt", keyfile="private.key")

# Wrap the server socket with SSL
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"Serving on https://localhost:{PORT}")
httpd.serve_forever()
