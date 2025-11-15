import { useState } from "react";
import { Settings, RotateCcw, Edit3, Save, X } from "lucide-react";
import { TrashCanData } from "./Dashboard";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

interface ControlPanelProps {
  data: TrashCanData;
  onUpdate: (updates: Partial<TrashCanData>) => void;
  onReset: () => void;
}

export function ControlPanel({ data, onUpdate, onReset }: ControlPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [editedData, setEditedData] = useState(data);

  const handleSave = () => {
    onUpdate(editedData);
    setIsOpen(false);
  };

  const handleCancel = () => {
    setEditedData(data);
    setIsOpen(false);
  };

  const updateCategory = (category: keyof typeof data.categories, value: number) => {
    setEditedData({
      ...editedData,
      categories: {
        ...editedData.categories,
        [category]: value
      }
    });
  };

  const categoryLabels = {
    recyclable: 'RECYCLABLE',
    organic: 'ORGANIC',
    plastic: 'PLASTIC',
    paper: 'PAPER',
    general: 'GENERAL'
  };

  return (
    <div className="bg-[#0f0f23] border-2 border-[#50d070]/30 p-4">
      <div className="mb-3">
        <span className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace' }}>
          === CONTROLS ===
        </span>
      </div>

      <div className="space-y-3">
        {/* Quick Target Category Change */}
        <div>
          <Label className="text-[#50d070]/70 tracking-wide mb-2 block" style={{ fontFamily: 'monospace' }}>
            TARGET CATEGORY:
          </Label>
          <Select
            value={data.targetCategory}
            onValueChange={(value) => onUpdate({ targetCategory: value as any })}
          >
            <SelectTrigger className="bg-[#1a1a3e] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide" style={{ fontFamily: 'monospace' }}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1a3e] border-2 border-[#50d070]">
              {Object.entries(categoryLabels).map(([key, label]) => (
                <SelectItem
                  key={key}
                  value={key}
                  className="text-[#50d070] tracking-wide focus:bg-[#0f0f23] focus:text-[#50d070]"
                  style={{ fontFamily: 'monospace' }}
                >
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Edit Data Button */}
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
          <DialogTrigger asChild>
            <Button
              className="w-full bg-[#1a1a3e] border-2 border-[#50d070] text-[#50d070] hover:bg-[#50d070] hover:text-[#0f0f23] tracking-wider transition-colors"
              style={{ fontFamily: 'monospace' }}
            >
              <Edit3 className="w-4 h-4 mr-2" />
              EDIT DATA
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-[#1a1a3e] border-4 border-[#50d070] text-[#50d070] max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="text-[#50d070] tracking-wider" style={{ fontFamily: 'monospace' }}>
                === EDIT TRASH CAN DATA ===
              </DialogTitle>
            </DialogHeader>

            <div className="space-y-4 mt-4">
              {/* Basic Info */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-[#50d070]/70 tracking-wide mb-2 block" style={{ fontFamily: 'monospace' }}>
                    FILL LEVEL (%):
                  </Label>
                  <Input
                    type="number"
                    min="0"
                    max="100"
                    value={editedData.fillLevel}
                    onChange={(e) => setEditedData({ ...editedData, fillLevel: parseInt(e.target.value) || 0 })}
                    className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                    style={{ fontFamily: 'monospace' }}
                  />
                </div>

                <div>
                  <Label className="text-[#50d070]/70 tracking-wide mb-2 block" style={{ fontFamily: 'monospace' }}>
                    WEIGHT (kg):
                  </Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="0"
                    value={editedData.weight}
                    onChange={(e) => setEditedData({ ...editedData, weight: parseFloat(e.target.value) || 0 })}
                    className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                    style={{ fontFamily: 'monospace' }}
                  />
                </div>
              </div>

              <div>
                <Label className="text-[#50d070]/70 tracking-wide mb-2 block" style={{ fontFamily: 'monospace' }}>
                  LOCATION:
                </Label>
                <Input
                  type="text"
                  value={editedData.location}
                  onChange={(e) => setEditedData({ ...editedData, location: e.target.value })}
                  className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                  style={{ fontFamily: 'monospace' }}
                />
              </div>

              {/* Categories */}
              <div className="border-t-2 border-[#50d070]/30 pt-4">
                <Label className="text-[#50d070] tracking-wider mb-3 block" style={{ fontFamily: 'monospace' }}>
                  CATEGORY COUNTS:
                </Label>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(categoryLabels).map(([key, label]) => (
                    <div key={key}>
                      <Label className="text-[#50d070]/70 tracking-wide mb-2 block" style={{ fontFamily: 'monospace' }}>
                        {label}:
                      </Label>
                      <Input
                        type="number"
                        min="0"
                        value={editedData.categories[key as keyof typeof editedData.categories]}
                        onChange={(e) => updateCategory(key as keyof typeof data.categories, parseInt(e.target.value) || 0)}
                        className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                        style={{ fontFamily: 'monospace' }}
                      />
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4">
                <Button
                  onClick={handleSave}
                  className="flex-1 bg-[#0f0f23] border-2 border-[#60d060] text-[#60d060] hover:bg-[#60d060] hover:text-[#0f0f23] tracking-wider"
                  style={{ fontFamily: 'monospace' }}
                >
                  <Save className="w-4 h-4 mr-2" />
                  SAVE
                </Button>
                <Button
                  onClick={handleCancel}
                  className="flex-1 bg-[#0f0f23] border-2 border-[#ff4466] text-[#ff4466] hover:bg-[#ff4466] hover:text-[#0f0f23] tracking-wider"
                  style={{ fontFamily: 'monospace' }}
                >
                  <X className="w-4 h-4 mr-2" />
                  CANCEL
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Reset Button */}
        <Button
          onClick={onReset}
          className="w-full bg-[#1a1a3e] border-2 border-[#ffaa44] text-[#ffaa44] hover:bg-[#ffaa44] hover:text-[#0f0f23] tracking-wider transition-colors"
          style={{ fontFamily: 'monospace' }}
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          RESET TO DEFAULT
        </Button>
      </div>
    </div>
  );
}
