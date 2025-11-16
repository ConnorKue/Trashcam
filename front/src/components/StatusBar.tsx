import { Activity } from "lucide-react";

interface StatusBarProps {
  totalCans: number;
  criticalCans: number;
  warningCans: number;
  currentTime: Date;
}

export function StatusBar({
  totalCans,
  criticalCans,
  warningCans,
  currentTime,
}: StatusBarProps) {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  return (
    <div className="flex flex-wrap items-center justify-center gap-4 mt-4">
      <div className="bg-[#1a1a3e] border-2 border-[#50d070] px-4 py-2 flex items-center gap-2">
        <Activity className="w-4 h-4 text-[#50d070]" />
        <span
          className="text-[#50d070] tracking-wider"
          style={{ fontFamily: "monospace" }}
        >
          ONLINE: {totalCans}
        </span>
      </div>

      {criticalCans > 0 && (
        <div className="bg-[#1a1a3e] border-2 border-[#ff4466] px-4 py-2 animate-pulse">
          <span
            className="text-[#ff4466] tracking-wider"
            style={{ fontFamily: "monospace" }}
          >
            CRITICAL: {criticalCans}
          </span>
        </div>
      )}

      {warningCans > 0 && (
        <div className="bg-[#1a1a3e] border-2 border-[#ffaa44] px-4 py-2">
          <span
            className="text-[#ffaa44] tracking-wider"
            style={{ fontFamily: "monospace" }}
          >
            WARNING: {warningCans}
          </span>
        </div>
      )}

      <div className="bg-[#1a1a3e] border-2 border-[#50d070]/50 px-4 py-2">
        <span
          className="text-[#50d070]/80 tracking-wider"
          style={{ fontFamily: "monospace" }}
        >
          {formatTime(currentTime)}
        </span>
      </div>
    </div>
  );
}
