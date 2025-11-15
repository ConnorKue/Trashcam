// frontend/src/App.jsx
import "./App.css";

const DEMO_STATS = {
  fillLevel: 72, // percent
  misassignedRate: 18, // percent
  totalDetections: 3456,
  lastUpdated: "12:03:45",
  typeBreakdown: {
    recyclable: 45,
    organic: 30,
    landfill: 25,
  },
  correctStreak: 12,
  sessionItems: 213,
  recentEvents: [
    { time: "12:03", type: "info", message: "apple → ORGANIC ✓" },
    { time: "12:01", type: "warn", message: "plastic bottle in LANDFILL" },
    { time: "11:58", type: "error", message: "camera dropped frames" },
    { time: "11:52", type: "info", message: "can → RECYCLABLE ✓" },
  ],
};

const formatPercent = (value) => `${Math.round(value ?? 0)}%`;

function App() {
  const health = "ok"; // hard-coded for now
  const stats = DEMO_STATS;
  const isOnline = health === "ok";

  return (
    <div className="app crt">
      <div className="scanline" />
      <div className="frame">
        <header className="header">
          <div>
            <h1>TRASHCAMTHINGY // CONTROL PANEL</h1>
            <p className="subtitle">SMART BIN STATUS MONITOR</p>
          </div>

          <div className="status-pill">
            <span className="dot" data-status={isOnline ? "ok" : "error"} />
            <span>{isOnline ? "ONLINE" : "OFFLINE"}</span>
          </div>
        </header>

        <main className="grid">
          {/* Fill Level Panel */}
          <section className="panel panel-main">
            <h2>FILL LEVEL</h2>
            <div className="fill-level">
              <div className="fill-bar">
                <div
                  className="fill-bar-inner"
                  style={{ width: `${stats.fillLevel ?? 0}%` }}
                />
              </div>
              <div className="fill-value">
                {formatPercent(stats.fillLevel)} FULL
              </div>
            </div>
            <div className="meta-row">
              <span>LAST UPDATE: {stats.lastUpdated || "—"}</span>
              <span>DETECTIONS: {stats.totalDetections ?? 0}</span>
            </div>
          </section>

          {/* Trash Type Mix */}
          <section className="panel">
            <h2>TRASH TYPE MIX</h2>
            <div className="type-list">
              {["recyclable", "organic", "landfill"].map((key) => (
                <TypeRow
                  key={key}
                  label={key.toUpperCase()}
                  value={stats.typeBreakdown?.[key] ?? 0}
                />
              ))}
            </div>
          </section>

          {/* Misassignment */}
          <section className="panel">
            <h2>MISASSIGNMENT RATE</h2>
            <div className="misassign">
              <div className="circle-gauge">
                <div className="circle-inner">
                  <span>{formatPercent(stats.misassignedRate ?? 0)}</span>
                  <small>MIS-ASSIGNED</small>
                </div>
              </div>
              <div className="misassign-info">
                <p>
                  TARGET &lt; <span>5%</span>
                </p>
                <p>
                  CURRENT STREAK:{" "}
                  <span>{stats.correctStreak ?? 0} ITEMS</span>
                </p>
                <p>
                  SESSION TOTAL: <span>{stats.sessionItems ?? 0}</span>
                </p>
              </div>
            </div>
          </section>

          {/* Recent Events / Log */}
          <section className="panel panel-log">
            <h2>EVENT LOG</h2>
            <div className="log">
              {(stats.recentEvents ?? []).length === 0 ? (
                <div className="log-empty">NO RECENT EVENTS</div>
              ) : (
                stats.recentEvents.slice(0, 8).map((evt, idx) => (
                  <div key={idx} className="log-row">
                    <span className="log-time">{evt.time}</span>
                    <span
                      className={`log-tag log-tag-${evt.type || "info"}`}
                    >
                      {(evt.type || "info").toUpperCase()}
                    </span>
                    <span className="log-text">{evt.message}</span>
                  </div>
                ))
              )}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

function TypeRow({ label, value }) {
  return (
    <div className="type-row">
      <div className="type-row-header">
        <span>{label}</span>
        <span className="type-row-value">{Math.round(value ?? 0)}%</span>
      </div>
      <div className="type-bar">
        <div
          className="type-bar-inner"
          style={{ width: `${value ?? 0}%` }}
        />
      </div>
    </div>
  );
}

export default App;
