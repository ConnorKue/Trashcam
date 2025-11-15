interface PixelProgressBarProps {
  value: number;
  color: string;
}

export function PixelProgressBar({ value, color }: PixelProgressBarProps) {
  const segments = 20;
  const filledSegments = Math.round((value / 100) * segments);

  return (
    <div className="flex gap-[2px]">
      {Array.from({ length: segments }).map((_, index) => (
        <div
          key={index}
          className="h-6 flex-1 border-2 border-[#0f0f23]"
          style={{
            backgroundColor: index < filledSegments ? color : '#1a1a3e',
            boxShadow: index < filledSegments ? `0 0 8px ${color}66` : 'none'
          }}
        />
      ))}
    </div>
  );
}
