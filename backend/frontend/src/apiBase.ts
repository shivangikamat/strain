/** Backend origin for API calls. Empty in dev so Vite proxies `/api` to localhost. */
const raw = (import.meta.env.VITE_API_BASE_URL ?? '').trim()
const base = raw.replace(/\/$/, '')

export function apiUrl(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`
  if (!base) return p
  return `${base}${p}`
}
