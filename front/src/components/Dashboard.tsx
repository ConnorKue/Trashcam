import { useState, useEffect } from "react";
import { TrashCanCard } from "./TrashCanCard";
import { StatusBar } from "./StatusBar";
import { EventLog } from "./EventLog";
import { Trash2 } from "lucide-react";
import { getLogs, getFill } from "../api";

export type EventType =
  | "deposit"
  | "empty"
  | "alert"
  | "contamination";

export interface TrashCanData {
  id: string;
  name: string;
  fillLevel: number;
  weight: number;
  lastEmptied: string; // ISO timestamp
  status: "normal" | "warning" | "critical";
  location: string;
  targetCategory: "recyclable" | "organic/compost" | "general";
  categories: {
    recyclable: number;
    organic: number;
    general: number;
  };
  events: Array<{
    timestamp: string;
    type: EventType;
    message: string;
  }>;
}

const initialTrashCanData: TrashCanData = {
  id: "TC001",
  name: "Main Entrance",
  fillLevel: 45,
  weight: 12.3,
  lastEmptied: new Date().toISOString(),
  status: "normal",
  location: "Building A",
  targetCategory: "recyclable",
  categories: {
    recyclable: 20,
    organic: 15,
    general: 10,
  },
  events: [],
};

interface BackendLog {
  timestamp: string;
  item: string;
  class: string;
}

export function Dashboard() {
  const [trashCans, setTrashCans] = useState<TrashCanData[]>([
    initialTrashCanData,
  ]);
  const [currentTime, setCurrentTime] = useState(new Date());

  const primaryTargetCategory = trashCans[0]?.targetCategory;

  const calculateStatus = (
    fillLevel: number
  ): "normal" | "warning" | "critical" => {
    if (fillLevel >= 90) return "critical";
    if (fillLevel >= 75) return "warning";
    return "normal";
  };

  // Clock
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Load logs & classify as deposit vs contamination based on targetCategory
  useEffect(() => {
    async function loadEvents() {
      try {
        const data = await getLogs();
        const logs = data.logs as BackendLog[];

        const detectionEvents = logs.map((log): TrashCanData["events"][number] => {
          const ts = log.timestamp;
          const cls = (log.class || "").toLowerCase();
          const itemName = log.item || "";
          const target = primaryTargetCategory?.toLowerCase();

          const isContamination =
            !!target && cls && cls !== target;

          const type: EventType = isContamination
            ? "contamination"
            : "deposit";

          const message = isContamination
            ? `Non-${primaryTargetCategory?.toUpperCase()} item detected (${itemName} – ${cls || "unknown"})`
            : `${cls || "item"} detected (${itemName})`;

          return {
            timestamp: ts,
            type,
            message,
          };
        });

        setTrashCans((prev) =>
          prev.map((can, index) => {
            if (index !== 0) return can;

            // keep manual events (empty, alert), replace only deposit/contamination
            const manualEvents = can.events.filter(
              (e) => e.type === "empty" || e.type === "alert"
            );

            return {
              ...can,
              events: [...manualEvents, ...detectionEvents],
            };
          })
        );
      } catch (err) {
        console.error("Error fetching events:", err);
      }
    }

    loadEvents();
    const id = setInterval(loadEvents, 2000); // poll logs
    return () => clearInterval(id);
  }, [primaryTargetCategory]);

  // Poll /fill and update fill level + status + add alert when >= 80%
  useEffect(() => {
    async function pollFill() {
      try {
        const data = await getFill();
        const fill = typeof data.fill === "number" ? data.fill : 0;

        setTrashCans((prev) =>
          prev.map((can, index) => {
            if (index !== 0) return can;

            const prevFill = can.fillLevel;
            const newStatus = calculateStatus(fill);
            let events = can.events;

            // Add a warning event only when crossing the 80% threshold
            if (fill >= 80 && prevFill < 80) {
              const now = new Date();
              const timeStr = now.toLocaleTimeString("en-US", {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              });

              const alertEvent: TrashCanData["events"][number] = {
                timestamp: timeStr,
                type: "alert",
                message: `Fill level reached ${fill}%`,
              };

              events = [alertEvent, ...events].slice(0, 50);
            }

            return {
              ...can,
              fillLevel: fill,
              status: newStatus,
              events,
            };
          })
        );
      } catch (err) {
        console.error("Error fetching fill level:", err);
      }
    }

    pollFill();
    const id = setInterval(pollFill, 5000); // every 5s
    return () => clearInterval(id);
  }, []);

  const updateTrashCan = (id: string, updates: Partial<TrashCanData>) => {
    setTrashCans((prev) =>
      prev.map((can) => {
        if (can.id !== id) return can;
        const updatedCan = { ...can, ...updates };
        if (updates.fillLevel !== undefined) {
          updatedCan.status = calculateStatus(updates.fillLevel);
        }
        return updatedCan;
      })
    );
  };

  const emptyTrashCan = (id: string) => {
    setTrashCans((prev) =>
      prev.map((can) => {
        if (can.id !== id) return can;

        const emptiedCategories = Object.keys(can.categories).reduce(
          (acc, key) => {
            acc[key as keyof typeof can.categories] = 0;
            return acc;
          },
          {} as typeof can.categories
        );

        const now = new Date();
        const currentTimeString = now.toLocaleTimeString("en-US", {
          hour12: false,
        });

        return {
          ...can,
          fillLevel: 0,
          weight: 0,
          categories: emptiedCategories,
          lastEmptied: now.toISOString(),
          status: "normal",
          events: [
            {
              timestamp: currentTimeString,
              type: "empty" as const,
              message: "Trash can emptied",
            },
            ...can.events.slice(0, 49),
          ],
        };
      })
    );
  };

  const removeEvent = (trashCanId: string, eventIndex: number) => {
    setTrashCans((prev) =>
      prev.map((can) => {
        if (can.id !== trashCanId) return can;

        const updatedEvents = [...can.events];
        updatedEvents.splice(eventIndex, 1);

        return {
          ...can,
          events: updatedEvents,
        };
      })
    );
  };

  const totalCans = trashCans.length;
  const criticalCans = trashCans.filter(
    (can) => can.status === "critical"
  ).length;
  const warningCans = trashCans.filter(
    (can) => can.status === "warning"
  ).length;

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8 text-center">
        <div className="inline-flex items-center justify-center gap-3 mb-4 px-6 py-3 bg-[#1a1a3e] border-4 border-[#50d070] shadow-[4px_4px_0px_0px_rgba(80,208,112,0.3)]">
          <Trash2 className="w-8 h-8 text-[#50d070]" />
          <h1
            className="text-[#50d070] tracking-wider"
            style={{
              fontFamily: "monospace",
              textShadow: "2px 2px 0px rgba(80,208,112,0.3)",
            }}
          >
            SMART TRASH MONITOR v1.0
          </h1>
        </div>
        <StatusBar
          totalCans={totalCans}
          criticalCans={criticalCans}
          warningCans={warningCans}
          currentTime={currentTime}
        />
      </div>

      {/* Grid of Trash Cans */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
        <div className="lg:col-span-2">
          {trashCans.map((trashCan) => (
            <TrashCanCard
              key={trashCan.id}
              data={trashCan}
              onUpdate={(updates) => updateTrashCan(trashCan.id, updates)}
              emptyTrash={() => emptyTrashCan(trashCan.id)}
              currentTime={currentTime}
            />
          ))}
        </div>
        <div className="lg:col-span-1">
          {trashCans.map((trashCan) => (
            <EventLog
              key={`events-${trashCan.id}`}
              events={trashCan.events}
              onRemoveEvent={(index) => removeEvent(trashCan.id, index)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
