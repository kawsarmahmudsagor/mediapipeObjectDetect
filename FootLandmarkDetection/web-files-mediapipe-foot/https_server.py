import http.server
import ssl

PORT = 5433
Handler = http.server.SimpleHTTPRequestHandler

httpd = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
httpd.socket = ssl.wrap_socket(httpd.socket,
                               keyfile="private.key",
                               certfile="certificate.crt",
                               server_side=True)

print(f"Serving on https://localhost:{PORT}")
httpd.serve_forever()
