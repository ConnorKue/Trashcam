const API_BASE_URL = "http://localhost:8000"; // hard-coded as you had

async function request(path: string, options: RequestInit = {}) {
  const url = `${API_BASE_URL}${path}`;
  console.log("[API] Fetching:", url, options);

  try {
    const res = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
      },
      ...options,
    });

    console.log("[API] Response status:", res.status, "for", url);

    if (!res.ok) {
      const text = await res.text().catch(() => "<no body>");
      console.error("[API] Error response body:", text);
      throw new Error(`Request failed: ${res.status}`);
    }

    return res.json();
  } catch (err) {
    console.error("[API] Fetch error for", url, err);
    throw err;
  }
}

export async function getHealth() {
  // If you still have /health/, use that.
  // If not used anywhere, this doesn't matter.
  return request("/health/");
}

export async function getLogs() {
  // FastAPI logs router: prefix="/logs", @router.get("/")
  // => GET /logs/
  return request("/logs/");
}

export async function getFill() {
  // Fill router: prefix="/fill", @router.get("/")
  // => GET /fill/
  const res = await request("/fill/");
  return res.fillPercent.toFixed(2);
}

export async function clearCurrentData() {
  // ClearData router: prefix="/clearData", @router.delete("/")
  // => DELETE /clearData/
  return request("/clearData/", {
    method: "DELETE",
  });
}
