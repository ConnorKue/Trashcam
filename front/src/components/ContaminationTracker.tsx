import { AlertTriangle, Target } from "lucide-react";

interface ContaminationTrackerProps {
  targetCategory: "recyclable" | "organic" | "general";
  categories: {
    recyclable: number;
    organic: number;
    general: number;
  };
}

export function ContaminationTracker({ targetCategory, categories }: ContaminationTrackerProps) {
  const categoryLabels = {
    recyclable: 'RECYCLABLE',
    organic: 'ORGANIC',
    general: 'GENERAL TRASH'
  };

  const categoryColors = {
    recyclable: '#50d0e0',
    organic: '#60d060',
    general: '#d050d0'
  };

  const targetCount = categories[targetCategory];
  const totalCount = Object.values(categories).reduce((sum, count) => sum + count, 0);
  const contaminationCount = totalCount - targetCount;
  const contaminationPercentage = totalCount > 0 ? Math.round((contaminationCount / totalCount) * 100) : 0;

  const targetColor = categoryColors[targetCategory];
  const isContaminated = contaminationCount > 0;

  return (
    <div
      className="bg-[#0f0f23] border-2 p-4 mb-4"
      style={{ borderColor: isContaminated ? '#ff4466' : '#50d070' }}
    >
      <div className="flex items-center gap-2 mb-3">
        <Target className="w-5 h-5" style={{ color: targetColor }} />
        <span className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace' }}>
          TARGET CATEGORY
        </span>
      </div>

      <div className="bg-[#1a1a3e] border-2 p-3 mb-3" style={{ borderColor: `${targetColor}66` }}>
        <p className="tracking-wider text-center" style={{ fontFamily: 'monospace', color: targetColor }}>
          {categoryLabels[targetCategory]}
        </p>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between">
          <span className="text-[#50d070]/70 tracking-wide" style={{ fontFamily: 'monospace' }}>
            Correct Items:
          </span>
          <span className="tracking-wider" style={{ fontFamily: 'monospace', color: targetColor }}>
            {targetCount}
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-[#50d070]/70 tracking-wide" style={{ fontFamily: 'monospace' }}>
            Contamination:
          </span>
          <span
            className="tracking-wider flex items-center gap-1"
            style={{ fontFamily: 'monospace', color: isContaminated ? '#ff4466' : '#60d060' }}
          >
            {isContaminated && <AlertTriangle className="w-3 h-3" />}
            {contaminationCount}
          </span>
        </div>

        <div className="pt-2 border-t border-[#50d070]/20">
          <div className="flex justify-between">
            <span className="text-[#50d070]/70 tracking-wide" style={{ fontFamily: 'monospace' }}>
              Contamination Rate:
            </span>
            <span
              className="tracking-wider"
              style={{
                fontFamily: 'monospace',
                color: contaminationPercentage > 30 ? '#ff4466' : contaminationPercentage > 10 ? '#ffaa44' : '#60d060'
              }}
            >
              {contaminationPercentage}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
