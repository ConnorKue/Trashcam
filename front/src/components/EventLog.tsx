import { useState } from "react";
import { ScrollArea } from "./ui/scroll-area";
import { Clock, Trash2, AlertTriangle, Package } from "lucide-react";

interface Event {
  timestamp: string;
  type: "deposit" | "empty" | "alert" | "contamination";
  message: string;
}

interface EventLogProps {
  events: Event[];
  scrollHeight?: number;
  onRemoveEvent?: (index: number) => void;
}

export function EventLog({ events, scrollHeight = 800, onRemoveEvent }: EventLogProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const getEventIcon = (type: string) => {
    switch (type) {
      case "deposit":
        return Package;
      case "empty":
        return Trash2;
      case "alert":
        return AlertTriangle;
      case "contamination":
        return AlertTriangle;
      default:
        return Clock;
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case "deposit":
        return "#50d0e0";
      case "empty":
        return "#60d060";
      case "alert":
        return "#ffaa44";
      case "contamination":
        return "#ff4466";
      default:
        return "#50d070";
    }
  };

  const handleRemoveEvent = (index: number) => {
    if (onRemoveEvent) {
      onRemoveEvent(index);
    }
  };

  return (
    <div className="bg-[#1a1a3e] border-4 border-[#50d070] p-4 shadow-[8px_8px_0px_0px_rgba(0,0,0,0.5)]">
      <div className="mb-4">
        <h2 className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace', textShadow: '2px 2px 0px rgba(80,208,112,0.3)' }}>
          === EVENT LOG ===
        </h2>
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
                className="bg-[#0f0f23] border-2 p-3 cursor-pointer transition-all duration-200 ease-in-out"
                style={{
                  borderColor: isHovered ? color : `${color}66`,
                  boxShadow: isHovered ? `0 0 8px ${color}80` : 'none'
                }}
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
                onClick={() => handleRemoveEvent(index)}
              >
                <div className="flex items-start gap-2">
                  <Icon className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color }} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className="tracking-wider"
                        style={{ fontFamily: 'monospace', color }}
                      >
                        {event.timestamp}
                      </span>
                    </div>
                    <p
                      className="tracking-wide break-words"
                      style={{ fontFamily: 'monospace', color: `${color}cc` }}
                    >
                      {event.message}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
