/** Shared HTTP/error helpers for talking to the FastAPI backend. */

/** FastAPI errors come back as { detail: string | {msg,loc}[] | object }. Render readable text. */
export function errMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msg = detail
      .map((d: any) => (d?.msg ? `${d.msg}${d.loc ? ` (${d.loc.join(".")})` : ""}` : JSON.stringify(d)))
      .filter(Boolean)
      .join("; ");
    return msg || fallback;
  }
  if (detail && typeof detail === "object") return (detail as any).msg ?? JSON.stringify(detail);
  return fallback;
}

export async function readError(res: Response, fallback: string): Promise<string> {
  const body = await res.json().catch(() => ({}));
  return errMessage((body as any)?.detail, fallback);
}
