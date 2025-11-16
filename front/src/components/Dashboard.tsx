import { useState, useEffect, useRef } from "react";
import { TrashCanCard } from "./TrashCanCard";
import { StatusBar } from "./StatusBar";
import { EventLog } from "./EventLog";
import { Trash2 } from "lucide-react";
import { getLogs, getFill } from "../api";

export type EventType = "deposit" | "empty" | "alert" | "contamination";

export interface TrashCanData {
  id: string;
  name: string;
  fillLevel: number;
  weight: number;
  lastEmptied: string; // ISO timestamp
  status: "normal" | "warning" | "critical";
  location: string;
  targetCategory: "recyclable" | "organic" | "general";
  categories: {
    recyclable: number;
    organic: number;
    general: number;
  };
  events: Array<{
    timestamp: string;
    type: EventType;
    category: string;
    message: string;
  }>;
}

const initialTrashCanData: TrashCanData = {
  id: "TC001",
  name: "Main Entrance",
  fillLevel: 0,
  weight: 0,
  lastEmptied: new Date().toISOString(),
  status: "normal",
  location: "Building A",
  targetCategory: "recyclable",
  categories: {
    recyclable: 0,
    organic: 0,
    general: 0,
  },
  events: [],
};

interface BackendLog {
  timestamp: string;
  item: string;
  class: string;
}

// Simple classification function for 3-category system
const classifyItem = (cls: string): "recyclable" | "organic" | "general" => {
  const c = cls.toLowerCase();
  if (c.includes("plastic") || c.includes("paper") || c.includes("cardboard") ||
      c.includes("metal") || c.includes("glass") || c.includes("can") ||
      c.includes("aluminum")) {
    return "recyclable";
  } else if (c.includes("organic") || c.includes("compost") || c.includes("food") ||
             c.includes("biodegradable") || c.includes("fruit") || c.includes("vegetable")) {
    return "organic";
  } else {
    return "general";
  }
};

export function Dashboard() {
  const [trashCans, setTrashCans] = useState<TrashCanData[]>([
    initialTrashCanData,
  ]);
  const [currentTime, setCurrentTime] = useState(new Date());

  const processedLogs = useRef<Set<string>>(new Set());
  const primaryTargetCategory = trashCans[0]?.targetCategory;

  const calculateStatus = (
    fillLevel: number
  ): "normal" | "warning" | "critical" => {
    if (fillLevel >= 90) return "critical";
    if (fillLevel >= 75) return "warning";
    return "normal";
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    setTrashCans((prev) =>
      prev.map((can) => ({
        ...can,
        events: can.events.map((event) => {
          if (event.type === "deposit" || event.type === "contamination") {
            const isContamination = event.category !== can.targetCategory;
            const newType: EventType = isContamination ? "contamination" : "deposit";
            const message = isContamination
              ? `Non-${can.targetCategory.toUpperCase()}: ${event.category} (${event.message.split('(')[1]?.split(' – ')[0] || "item"})`
              : `${event.category} detected (${event.message.split('(')[1]?.split(' – ')[0] || "item"})`;

            return {
              ...event,
              type: newType,
              message,
            };
          }
          return event;
        }),
      }))
    );
  }, [trashCans.map(can => can.targetCategory).join()]);

  // Load logs & classify as deposit vs contamination based on targetCategory
  useEffect(() => {
    async function loadEvents() {
      try {
        const [logsData, fillData] = await Promise.all([
          getLogs().catch((err) => {
            console.error("getLogs failed:", err);
            return null;
          }),
          getFill().catch((err) => {
            console.error("getFill failed:", err);
            return 0;
          }),
        ]);
        const logs = logsData.logs as BackendLog[];
        const fill = fillData;
        setTrashCans((prev) =>
          prev.map((can, index) => {
            if (index !== 0) return can;

            let updatedCategories = { ...can.categories };
            let changes = 0;
            const newEvents: TrashCanData["events"][] = [];
            let hasNewItems = false;

            // Process each log to update categories and create events
            logs.forEach((log) => {
              const logSignature = `${log.timestamp}-${log.item}-${log.class}`;
              // Skip if we've already processed this log
              if (processedLogs.current.has(logSignature)) {
                return;
              }

              processedLogs.current.add(logSignature);
              hasNewItems = true;

              const detectedCategory = classifyItem(log.class);
              const isContamination = detectedCategory !== can.targetCategory;

              // update the category counter
              updatedCategories[detectedCategory] += 1;
              changes += 1;

              // create event
              const eventType: EventType = isContamination ? "contamination" : "deposit";
              const message = isContamination
                ? `Non-${can.targetCategory.toUpperCase()}: ${detectedCategory} (${log.item} – ${log.class || "unknown"})`
                : `${log.class || "item"} detected (${log.item})`;

              newEvents.push({
                timestamp: log.timestamp,
                type: eventType,
                category: detectedCategory,
                message,
              });
            });

            const newWeight = can.weight + Math.min(25, changes * 0.5);
            const newStatus = calculateStatus(fill);
            const existingEvents = [...can.events];

            // check if we need to add a fill level alert
            let finalEvents = [...newEvents, ...existingEvents];
            const prevFill = can.fillLevel;
            if (fill >= 80 && prevFill < 80) {
              const now = new Date();
              const timeStr = now.toLocaleTimeString("en-US", {
                hour12: false,
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              });

              finalEvents.unshift({
                timestamp: timeStr,
                type: "alert",
                message: `Fill level reached ${fill}%`,
              });
            }

            // limit events to last 50
            finalEvents = finalEvents.slice(0, 50);

            return {
              ...can,
              categories: updatedCategories,
              fillLevel: fill,
              weight: parseFloat(newWeight.toFixed(1)),
              status: newStatus,
              events: finalEvents,
            };
          })
        );
      } catch (err) {
        console.error("Error fetching data:", err);
      }
    }

    loadEvents();
    const id = setInterval(loadEvents, 2000); // poll logs
    return () => clearInterval(id);
  }, [primaryTargetCategory]);

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

        processedLogs.current.clear();

        const now = new Date();
        const currentTimeString = now.toLocaleTimeString("en-US", {
          hour12: false,
        });

        return {
          ...can,
          fillLevel: 0,
          weight: 0,
          categories: {
            recyclable: 0,
            organic: 0,
            general: 0,
          },
          lastEmptied: now.toISOString(),
          status: "normal",
          events: [
            {
              timestamp: currentTimeString,
              type: "empty" as const,
              category: "system",
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
              targetCategory={trashCan.targetCategory}
              onRemoveEvent={(index) => removeEvent(trashCan.id, index)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
