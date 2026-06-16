export function getServerOrigin(): string {
  if (import.meta.env.VITE_SOCKET_URL) {
    return import.meta.env.VITE_SOCKET_URL;
  }

  if (typeof window === 'undefined') {
    return 'http://localhost:8080';
  }

  const { protocol, hostname, port, origin } = window.location;

  if (port === '5173') {
    return `${protocol}//${hostname}:8080`;
  }

  return origin;
}
