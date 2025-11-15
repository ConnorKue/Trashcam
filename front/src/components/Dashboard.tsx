import { useState, useEffect } from "react";
import { TrashCanCard } from "./TrashCanCard";
import { StatusBar } from "./StatusBar";
import { EventLog } from "./EventLog";
import { Trash2 } from "lucide-react";

export interface TrashCanData {
  id: string;
  name: string;
  fillLevel: number;
  weight: number;
  lastEmptied: string;
  status: "normal" | "warning" | "critical";
  location: string;
  targetCategory: "recyclable" | "organic" | "plastic" | "paper" | "general";
  categories: {
    recyclable: number;
    organic: number;
    plastic: number;
    paper: number;
    general: number;
  };
  events: Array<{
    timestamp: string;
    type: "deposit" | "empty" | "alert" | "contamination";
    message: string;
  }>;
}

export function Dashboard() {
  const [trashCans, setTrashCans] = useState<TrashCanData[]>([
    {
      id: "TC001",
      name: "Main Entrance",
      fillLevel: 45,
      weight: 12.3,
      lastEmptied: "2 hours ago",
      status: "normal",
      location: "Building A",
      targetCategory: "recyclable",
      categories: {
        recyclable: 8,
        organic: 3,
        plastic: 12,
        paper: 15,
        general: 7
      },
      events: [
        { timestamp: "14:23:45", type: "deposit", message: "Paper item detected" },
        { timestamp: "14:18:32", type: "contamination", message: "Non-recyclable item detected" },
        { timestamp: "14:15:10", type: "deposit", message: "Plastic bottle detected" },
        { timestamp: "14:12:03", type: "deposit", message: "Recyclable can detected" },
        { timestamp: "12:30:00", type: "empty", message: "Trash can emptied" },
        { timestamp: "12:15:22", type: "alert", message: "Fill level reached 75%" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
        { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" },
      ]
    }
  ]);

  const defaultTrashCanData: TrashCanData = {
    id: "TC001",
    name: "Main Entrance",
    fillLevel: 45,
    weight: 12.3,
    lastEmptied: "2 hours ago",
    status: "normal",
    location: "Building A",
    targetCategory: "recyclable",
    categories: {
      recyclable: 8,
      organic: 3,
      plastic: 12,
      paper: 15,
      general: 7
    },
    events: [
      { timestamp: "14:23:45", type: "deposit", message: "Paper item detected" },
      { timestamp: "14:18:32", type: "contamination", message: "Non-recyclable item detected" },
      { timestamp: "14:15:10", type: "deposit", message: "Plastic bottle detected" },
      { timestamp: "14:12:03", type: "deposit", message: "Recyclable can detected" },
      { timestamp: "12:30:00", type: "empty", message: "Trash can emptied" },
      { timestamp: "12:15:22", type: "alert", message: "Fill level reached 75%" },
      { timestamp: "11:45:18", type: "deposit", message: "Organic waste detected" }
    ]
  };

  const updateTrashCan = (updates: Partial<TrashCanData>) => {
    setTrashCans([{ ...trashCans[0], ...updates }]);
  };

  const resetTrashCan = () => {
    setTrashCans([defaultTrashCanData]);
  };

  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const totalCans = trashCans.length;
  const criticalCans = trashCans.filter(can => can.status === "critical").length;
  const warningCans = trashCans.filter(can => can.status === "warning").length;

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8 text-center">
        <div className="inline-flex items-center justify-center gap-3 mb-4 px-6 py-3 bg-[#1a1a3e] border-4 border-[#50d070] shadow-[4px_4px_0px_0px_rgba(80,208,112,0.3)]">
          <Trash2 className="w-8 h-8 text-[#50d070]" />
          <h1 className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace', textShadow: '2px 2px 0px rgba(80,208,112,0.3)' }}>
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
              onUpdate={updateTrashCan}
              onReset={resetTrashCan}
            />
          ))}
        </div>
        <div className="lg:col-span-1">
          {trashCans.map((trashCan) => (
            <EventLog key={`events-${trashCan.id}`} events={trashCan.events} />
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 text-center">
        <div className="inline-block px-4 py-2 bg-[#1a1a3e] border-2 border-[#50d070]/30">
          <p className="text-[#50d070]/70 tracking-wide" style={{ fontFamily: 'monospace' }}>
            [ SYSTEM ONLINE ] PRESS F5 TO REFRESH
          </p>
        </div>
      </div>
    </div>
  );
}
