import { useState, useEffect } from "react";
import { RotateCcw, Edit3, Trash2, Save, X } from "lucide-react";
import { TrashCanData } from "./Dashboard";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogDescription,
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
import { clearCurrentData } from "../api";

interface ControlPanelProps {
  data: TrashCanData;
  onUpdate: (updates: Partial<TrashCanData>) => void;
  currentTime: Date;
  emptyTrash: () => void;
  clearEvents: () => void;
}

export function ControlPanel({
  data,
  onUpdate,
  emptyTrash,
  clearEvents,
}: ControlPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [editedData, setEditedData] = useState(data);

  useEffect(() => {
    if (!isOpen) {
      setEditedData(data);
    }
  }, [data, isOpen]);

  const handleSave = () => {
    const total = Object.values(data.categories).reduce(
      (sum, count) => sum + count,
      0
    );
    const calculatedWeight = Math.min(25, total * 0.5);
    const updates = {
      ...editedData,
      weight: parseFloat(calculatedWeight.toFixed(1)),
    };
    onUpdate(updates);
    setIsOpen(false);
  };

  const handleCancel = () => {
    setEditedData(data);
    setIsOpen(false);
  };

  const updateCategory = (
    category: keyof typeof data.categories,
    value: number
  ) => {
    const newValue = Math.max(0, value);
    setEditedData((prev) => ({
      ...prev,
      categories: {
        ...prev.categories,
        [category]: newValue,
      },
    }));
  };

  // Only two target modes now
  const targetModeOptions = {
    recyclable: "RECYCLING MODE",
    organic: "ORGANIC MODE",
    general: "TRASH MODE",
  } as const;

  const categoryLabels = {
    recyclable: "RECYCLABLE",
    organic: "ORGANIC",
    general: "GENERAL TRASH",
  };

  const handleEmptyTrashClick = async () => {
    try {
      await clearCurrentData(); // DELETE /clearData -> reset backend file
    } catch (err) {
      console.error("Failed to clear backend data:", err);
    }
    emptyTrash();
    clearEvents();
  };

  return (
    <div className="bg-[#0f0f23] border-2 border-[#50d070]/30 p-4">
      <div className="mb-3">
        <span
          className="text-[#50d070] tracking-wider"
          style={{ fontFamily: "monospace" }}
        >
          === CONTROLS ===
        </span>
      </div>

      <div className="space-y-3">
        {/* Target Mode Change */}
        <div>
          <Label
            className="text-[#50d070]/70 tracking-wide mb-2 block"
            style={{ fontFamily: "monospace" }}
          >
            TARGET MODE:
          </Label>
          <Select
            value={data.targetCategory}
            onValueChange={(value) =>
              onUpdate({
                targetCategory: value as "recyclable" | "organic" | "general",
              })
            }
          >
            <SelectTrigger
              className="bg-[#1a1a3e] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
              style={{ fontFamily: "monospace" }}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1a3e] border-2 border-[#50d070]">
              {Object.entries(targetModeOptions).map(([key, label]) => (
                <SelectItem
                  key={key}
                  value={key}
                  className="text-[#50d070] tracking-wide focus:bg-[#0f0f23] focus:text-[#50d070]"
                  style={{ fontFamily: "monospace" }}
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
              style={{ fontFamily: "monospace" }}
            >
              <Edit3 className="w-4 h-4 mr-2" />
              EDIT DATA
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-[#1a1a3e] border-4 border-[#50d070] text-[#50d070] max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle
                className="text-[#50d070] tracking-wider"
                style={{ fontFamily: "monospace" }}
              >
                === EDIT TRASH CAN DATA ===
              </DialogTitle>
              <DialogDescription className="sr-only">
                Edit trash can properties including fill level, weight,
                categories, and location information.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 mt-4">
              {/* Basic Info */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label
                    className="text-[#50d070]/70 tracking-wide mb-2 block"
                    style={{ fontFamily: "monospace" }}
                  >
                    FILL LEVEL (%):
                  </Label>
                  <Input
                    type="number"
                    min="0"
                    max="100"
                    value={editedData.fillLevel}
                    onChange={(e) =>
                      setEditedData({
                        ...editedData,
                        fillLevel: parseInt(e.target.value) || 0,
                      })
                    }
                    className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                    style={{ fontFamily: "monospace" }}
                  />
                </div>

                <div>
                  <Label
                    className="text-[#50d070]/70 tracking-wide mb-2 block"
                    style={{ fontFamily: "monospace" }}
                  >
                    WEIGHT (kg):
                  </Label>
                  <Input
                    type="number"
                    step="0.1"
                    min="0"
                    value={editedData.weight}
                    onChange={(e) =>
                      setEditedData({
                        ...editedData,
                        weight: parseFloat(e.target.value) || 0,
                      })
                    }
                    className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                    style={{ fontFamily: "monospace" }}
                  />
                </div>
              </div>

              <div>
                <Label
                  className="text-[#50d070]/70 tracking-wide mb-2 block"
                  style={{ fontFamily: "monospace" }}
                >
                  NAME:
                </Label>
                <Input
                  type="text"
                  value={editedData.name}
                  onChange={(e) =>
                    setEditedData({ ...editedData, name: e.target.value })
                  }
                  className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                  style={{ fontFamily: "monospace" }}
                />
              </div>

              <div>
                <Label
                  className="text-[#50d070]/70 tracking-wide mb-2 block"
                  style={{ fontFamily: "monospace" }}
                >
                  ID:
                </Label>
                <Input
                  type="text"
                  value={editedData.id}
                  onChange={(e) =>
                    setEditedData({ ...editedData, id: e.target.value })
                  }
                  className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                  style={{ fontFamily: "monospace" }}
                />
              </div>

              <div>
                <Label
                  className="text-[#50d070]/70 tracking-wide mb-2 block"
                  style={{ fontFamily: "monospace" }}
                >
                  LOCATION:
                </Label>
                <Input
                  type="text"
                  value={editedData.location}
                  onChange={(e) =>
                    setEditedData({
                      ...editedData,
                      location: e.target.value,
                    })
                  }
                  className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                  style={{ fontFamily: "monospace" }}
                />
              </div>

              {/* Categories */}
              <div className="border-t-2 border-[#50d070]/30 pt-4">
                <Label
                  className="text-[#50d070] tracking-wider mb-3 block"
                  style={{ fontFamily: "monospace" }}
                >
                  CATEGORY COUNTS:
                </Label>
                <div className="grid grid-cols-1 gap-3">
                  {Object.entries(categoryLabels).map(([key, label]) => (
                    <div key={key}>
                      <Label
                        className="text-[#50d070]/70 tracking-wide mb-2 block"
                        style={{ fontFamily: "monospace" }}
                      >
                        {label}:
                      </Label>
                      <Input
                        type="number"
                        min="0"
                        value={
                          editedData.categories[
                            key as keyof typeof editedData.categories
                          ]
                        }
                        onChange={(e) =>
                          updateCategory(
                            key as keyof typeof data.categories,
                            parseInt(e.target.value) || 0
                          )
                        }
                        className="bg-[#0f0f23] border-2 border-[#50d070]/50 text-[#50d070] tracking-wide"
                        style={{ fontFamily: "monospace" }}
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
                  style={{ fontFamily: "monospace" }}
                >
                  <Save className="w-4 h-4 mr-2" />
                  SAVE
                </Button>
                <Button
                  onClick={handleCancel}
                  className="flex-1 bg-[#0f0f23] border-2 border-[#ff4466] text-[#ff4466] hover:bg-[#ff4466] hover:text-[#0f0f23] tracking-wider"
                  style={{ fontFamily: "monospace" }}
                >
                  <X className="w-4 h-4 mr-2" />
                  CANCEL
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Empty Trash Button */}
        <Button
          onClick={handleEmptyTrashClick}
          className="w-full bg-[#1a1a3e] border-2 border-[#50d070] text-[#50d070] hover:bg-[#50d070] hover:text-[#0f0f23] tracking-wider transition-colors"
          style={{ fontFamily: "monospace" }}
        >
          <Trash2 className="w-4 h-4 mr-2" />
          EMPTY TRASH
        </Button>
      </div>
    </div>
  );
}
