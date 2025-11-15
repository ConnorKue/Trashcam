import { MapPin, Weight, Clock, AlertTriangle } from "lucide-react";
import { TrashCanData } from "./Dashboard";
import { PixelProgressBar } from "./PixelProgressBar";
import { CategoryBreakdown } from "./CategoryBreakdown";
import { ContaminationTracker } from "./ContaminationTracker";
import { ControlPanel } from "./ControlPanel";

interface TrashCanCardProps {
  data: TrashCanData;
  onUpdate: (id: string, fillLevel: number) => void;
  onReset: (id: string) => void;
}

export function TrashCanCard({ data, onUpdate, onReset }: TrashCanCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "critical":
        return "#ff4466";
      case "warning":
        return "#ffaa44";
      default:
        return "#50d070";
    }
  };

  const statusColor = getStatusColor(data.status);
  const borderColor = statusColor;

  return (
    <div
      //  transition-all hover:-translate-y-1
      className="bg-[#1a1a3e] border-4 p-6 shadow-[8px_8px_0px_0px_rgba(0,0,0,0.5)] hover:shadow-[12px_12px_0px_0px_rgba(0,0,0,0.5)]"
      style={{ borderColor }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2
            className="tracking-wider mb-1"
            style={{
              fontFamily: 'monospace',
              color: statusColor,
              textShadow: `2px 2px 0px ${statusColor}33`
            }}
          >
            {data.name}
          </h2>
          <p className="text-[#50d070]/60 tracking-wide" style={{ fontFamily: 'monospace' }}>
            ID: {data.id}
          </p>
        </div>
        {data.status === "critical" && (
          <div className="animate-pulse">
            <AlertTriangle className="w-6 h-6" style={{ color: statusColor }} />
          </div>
        )}
      </div>

      {/* Fill Level */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[#50d070]/80 tracking-wide" style={{ fontFamily: 'monospace' }}>
            FILL LEVEL
          </span>
          <span
            className="tracking-wider"
            style={{
              fontFamily: 'monospace',
              color: statusColor
            }}
          >
            {data.fillLevel}%
          </span>
        </div>
        <PixelProgressBar value={data.fillLevel} color={statusColor} />
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-[#0f0f23] border-2 border-[#50d070]/30 p-3">
          <div className="flex items-center gap-2 mb-1">
            <Weight className="w-4 h-4 text-[#50d070]" />
            <span className="text-[#50d070]/70 tracking-wide" style={{ fontFamily: 'monospace' }}>
              WEIGHT
            </span>
          </div>
          <p className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace' }}>
            {data.weight} kg
          </p>
        </div>

        <div className="bg-[#0f0f23] border-2 border-[#50d070]/30 p-3">
          <div className="flex items-center gap-2 mb-1">
            <Clock className="w-4 h-4 text-[#50d070]" />
            <span className="text-[#50d070]/70 tracking-wide" style={{ fontFamily: 'monospace' }}>
              EMPTIED
            </span>
          </div>
          <p className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace' }}>
            {data.lastEmptied}
          </p>
        </div>
      </div>

      {/* Location */}
      <div className="flex items-center gap-2 pt-3 border-t-2 border-[#50d070]/20 mb-4">
        <MapPin className="w-4 h-4 text-[#50d070]/70" />
        <span className="text-[#50d070]/70 tracking-wide" style={{ fontFamily: 'monospace' }}>
          {data.location}
        </span>
      </div>

      {/* Contamination Tracker */}
      <ContaminationTracker
        targetCategory={data.targetCategory}
        categories={data.categories}
      />

      {/* Category Breakdown */}
      <CategoryBreakdown categories={data.categories} />

      {/* Control Panel */}
      <ControlPanel
        data={data}
        onUpdate={onUpdate}
        onReset={onReset}
      />

    </div>
  );
}
