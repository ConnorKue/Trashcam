from fastapi import APIRouter

router = APIRouter(
    prefix="/percents",
    tags=["percents"]
)

@router.get("/CorrectPercent")
def PercentCorrect(State: str = ""):
    def read_lines_from(filepath, start_line):
        lines = []
        with open(filepath, 'r') as f:
            for current_line_number, line in enumerate(f):
                if current_line_number >= start_line:
                    lines.append(line.strip())
        return lines

    file_path = "../../logs.csv"
    log_lines = read_lines_from(file_path, 0)


    for i in range(len(log_lines)):
        log_lines[i] = log_lines[i].split(",")
        
    output = []
    Correct = 0
    Incorect = 0 
    for log in log_lines:
        if State == "" or log[3] == State:
            Correct += 1   
        else:
            Incorect += 1

    
    return {"Percent": Correct / (Correct + Incorect) * 100}