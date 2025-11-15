from fastapi import APIRouter

router = APIRouter(
    prefix="/totalTrash",
    tags=["toatlTrash"]
)


@router.get("/")
def TrashNumber():
    def count_lines_readlines(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return len(lines)

    file_path = "../../total_trash.csv"  
    num_lines = count_lines_readlines(file_path)