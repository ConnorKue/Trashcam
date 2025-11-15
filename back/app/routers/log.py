from fastapi import APIRouter

router = APIRouter(
    prefix="/logs",
    tags=["Logs"]
)


@router.get("/")
def LogOutput(State: str = "", curLine: int = 0):
    def read_lines_from(filepath, start_line):
        lines = []
        with open(filepath, 'r') as f:
            for current_line_number, line in enumerate(f):
                if current_line_number >= start_line:
                    lines.append(line.strip())
        return lines

    file_path = "../../logs.csv"
    log_lines = read_lines_from(file_path, curLine)


    for i in range(len(log_lines)):
        log_lines[i] = log_lines[i].split(",")
        
    output = []
    for log in log_lines:
        if State == "" or log[3] == State:
            output.append(log[0] + "Correct" + log[3])    
        else:
            output.append(log[0] + "Incorrect" + log[3])

    
    return {"logs": output}
    
