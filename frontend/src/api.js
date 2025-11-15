// frontend/src/api.js
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health/`);
  if (!res.ok) {
    throw new Error("Failed to fetch health");
  }
  return res.json();
}
