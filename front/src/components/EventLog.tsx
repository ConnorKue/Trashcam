import { useEffect, useState } from "react";
import { ScrollArea } from "./ui/scroll-area";
import { Package, AlertTriangle, Trash2, Recycle } from "lucide-react";
import { getLogs } from "../api";

interface Event {
  timestamp: string;
  type: "deposit" | "empty" | "alert" | "contamination";
  message: string;
}

interface EventLogProps {
  events: Event[];
  onRemoveEvent?: (index: number) => void;
  onClearAll?: () => void;
  scrollHeight?: number;
}

export function EventLog({
  events,
  onRemoveEvent,
  onClearAll,
  scrollHeight = 800,
}: EventLogProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const getEventIcon = (type: Event["type"]) => {
    switch (type) {
      case "deposit":
        return Package;
      case "contamination":
        return AlertTriangle;
      case "empty":
        return Trash2;
      case "alert":
        return AlertTriangle;
      default:
        return Package;
    }
  };

  const getEventColor = (type: Event["type"]) => {
    switch (type) {
      case "deposit":
        return "#50d0e0";
      case "contamination":
        return "#ff4466";
      case "empty":
        return "#50d070";
      case "alert":
        return "#ffaa44";
      default:
        return "#50d0e0";
    }
  };

  return (
    <div className="bg-[#1a1a3e] border-4 border-[#50d070] p-4 shadow-[8px_8px_0px_0px_rgba(0,0,0,0.5)]">
      <div className="mb-4">
        <div className="flex justify-between items-center">
          <h2
            className="text-[#50d070] tracking-wider"
            style={{
              fontFamily: "monospace",
              textShadow: "2px 2px 0px rgba(80,208,112,0.3)",
            }}
          >
            === EVENT LOG ===
          </h2>
          {onClearAll && events.length > 0 && (
            <button
              onClick={onClearAll}
              className="flex items-center gap-2 px-3 py-1 text-xs text-[#ff4466] hover:text-[#ff6688] border border-[#ff4466] hover:border-[#ff6688] bg-[#0f0f23] hover:bg-[#1a1a2e] transition-all duration-200 tracking-wide"
              style={{ fontFamily: "monospace" }}
              title="Clear all events"
            >
              <Recycle className="w-3 h-3" />
              CLEAR ALL
            </button>
          )}
        </div>
        <p
          className="text-xs text-[#50d070]/70 tracking-wide"
          style={{ fontFamily: "monospace" }}
        >
          LIVE DETECTION EVENTS
        </p>
      </div>

      <ScrollArea className="pr-4" style={{ height: scrollHeight }}>
        <div className="space-y-2">
          {events.map((event, index) => {
            const Icon = getEventIcon(event.type);
            const color = getEventColor(event.type);
            const isHovered = hoveredIndex === index;

            return (
              <div
                key={index}
                className="bg-[#0f0f23] border-2 p-3 cursor-pointer transition-all duration-200 ease-in-out group"
                style={{
                  borderColor: isHovered ? color : `${color}66`,
                  boxShadow: isHovered ? `0 0 8px ${color}80` : "none",
                }}
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
              >
                <div className="flex items-start gap-2">
                  <Icon
                    className="w-4 h-4 mt-0.5 flex-shrink-0"
                    style={{ color }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start">
                      <span
                        className="tracking-wider text-xs"
                        style={{ fontFamily: "monospace", color }}
                      >
                        {event.timestamp}
                      </span>
                      {onRemoveEvent && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onRemoveEvent(index);
                          }}
                          className="opacity-0 group-hover:opacity-100 text-[#ff4466] hover:text-[#ff6688] transition-opacity"
                        >
                          ×
                        </button>
                      )}
                    </div>
                    <p
                      className="tracking-wide break-words mt-1"
                      style={{ fontFamily: "monospace", color: `${color}cc` }}
                    >
                      {event.message}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}

          {events.length === 0 && (
            <div className="text-center py-8">
              <Package className="w-12 h-12 text-[#50d070]/80 mx-auto mb-2" />
              <p
                className="text-[#50d070]/80 text-sm tracking-wide"
                style={{ fontFamily: "monospace" }}>
                No events detected yet.
              </p>
              <p
                className="text-[#50d070]/80 text-xs tracking-wide mt-1"
                style={{ fontFamily: "monospace" }}>
                Items will appear here when detected.
              </p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
