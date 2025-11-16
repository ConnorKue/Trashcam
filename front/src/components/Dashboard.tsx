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
    plastic: number;
    paper: number;
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
  fillLevel: 0,
  weight: 0,
  lastEmptied: new Date().toISOString(),
  status: "normal",
  location: "Building A",
  targetCategory: "recycling",
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

  // calculate fill level based on item count (assuming 50 items = 100% fill)
  const calculateFillLevel = (categories: TrashCanData['categories']): number => {
    const totalItems = Object.values(categories).reduce((sum, count) => sum + count, 0);
    return Math.min(100, Math.round((totalItems / 50) * 100));
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

        setTrashCans((prev) =>
          prev.map((can, index) => {
            if (index !== 0) return can;

            let updatedCategories = { ...can.categories };
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

              // create event
              const eventType: EventType = isContamination ? "contamination" : "deposit";
              const message = isContamination
                ? `Non-${can.targetCategory.toUpperCase()}: ${detectedCategory} (${log.item} – ${log.class || "unknown"})`
                : `${log.class || "item"} detected (${log.item})`;

              newEvents.push({
                timestamp: log.timestamp,
                type: eventType,
                message,
              });
            });

            if (!hasNewItems) {
              return can;
            }

            // calculate new fill level based on updated categories
            const newFillLevel = calculateFillLevel(updatedCategories);
            const newStatus = calculateStatus(newFillLevel);

            // keep manual events and add new detection events
            const manualEvents = can.events.filter(
              (e) => e.type === "empty" || e.type === "alert"
            );

            // check if we need to add a fill level alert
            let finalEvents = [...manualEvents, ...newEvents];
            const prevFill = can.fillLevel;

            if (newFillLevel >= 80 && prevFill < 80) {
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
                message: `Fill level reached ${newFillLevel}%`,
              });
            }

            // limit events to last 50
            finalEvents = finalEvents.slice(0, 50);

            return {
              ...can,
              categories: updatedCategories,
              fillLevel: newFillLevel,
              status: newStatus,
              events: finalEvents,
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

  const updateTrashCan = (id: string, updates: Partial<TrashCanData>) => {
    setTrashCans((prev) =>
      prev.map((can) => {
        if (can.id !== id) return can;
        const updatedCan = { ...can, ...updates };
        if (updates.categories) {
          // Recalculate fill level if categories changed
          updatedCan.fillLevel = calculateFillLevel(updatedCan.categories);
          updatedCan.status = calculateStatus(updatedCan.fillLevel);
        } else if (updates.fillLevel !== undefined) {
          updatedCan.status = calculateStatus(updates.fillLevel);
        }
        return updatedCan;
      })
    );
  };

  const resetTrashCan = (id: string) => {
    setTrashCans((prev) =>
      prev.map((can) =>
        can.id === id
          ? {
              ...initialTrashCanData,
              id: can.id,
              name: can.name,
              location: can.location,
            }
          : can
      )
    );
  };

  const emptyTrashCan = (id: string) => {
    setTrashCans((prev) =>
      prev.map((can) => {
        if (can.id !== id) return can;

        processedLogs.current.clear();

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
              onReset={() => resetTrashCan(trashCan.id)}
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
