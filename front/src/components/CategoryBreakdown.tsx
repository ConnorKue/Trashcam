import { Recycle, Leaf, Package, FileText, Trash2 } from "lucide-react";

interface CategoryBreakdownProps {
  categories: {
    recyclable: number;
    organic: number;
    general: number;
  };
}

export function CategoryBreakdown({ categories }: CategoryBreakdownProps) {
  const categoryConfig = [
    { key: 'recyclable', label: 'RECYCLABLE', icon: Recycle, color: '#50d0e0' },
    { key: 'organic', label: 'ORGANIC', icon: Leaf, color: '#60d060' },
    { key: 'general', label: 'GENERAL TRASH', icon: Trash2, color: '#d050d0' }
  ];
  // purple: #d050d0
  // orange: #d0a040

  const total = Object.values(categories).reduce((sum, count) => sum + count, 0);

  const getPercentage = (count: number) => {
    if (total === 0) return 0;
    return Math.round((count / total) * 100);
  };

  return (
    <div className="bg-[#0f0f23] border-2 border-[#50d070]/30 p-4 mb-4">
      <div className="mb-3">
        <span className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace' }}>
          === TRASH CATEGORIES ===
        </span>
      </div>

      <div className="space-y-2">
        {categoryConfig.map(({ key, label, icon: Icon, color }) => {
          const count = categories[key as keyof typeof categories];
          const percentage = getPercentage(count);
          return (
            <div key={key} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Icon className="w-4 h-4" style={{ color }} />
                <span
                  className="tracking-wide"
                  style={{ fontFamily: 'monospace', color: `${color}cc` }}
                >
                  {label}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex gap-[2px]">
                  {Array.from({ length: Math.min(count, 10) }).map((_, i) => (
                    <div
                      key={i}
                      className="w-2 h-4 border border-[#0f0f23]"
                      style={{
                        backgroundColor: color,
                        boxShadow: `0 0 4px ${color}66`
                      }}
                    />
                  ))}
                </div>
                <span
                  className="tracking-wider min-w-[3rem] text-right"
                  style={{ fontFamily: 'monospace', color }}
                >
                  {count} ({percentage}%)
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-3 pt-3 border-t-2 border-[#50d070]/20">
        <div className="flex justify-between">
          <span className="text-[#50d070]/70 tracking-wide" style={{ fontFamily: 'monospace' }}>
            TOTAL ITEMS:
          </span>
          <span className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace' }}>
            {total}
          </span>
        </div>
      </div>
    </div>
  );
}
