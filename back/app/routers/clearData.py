from fastapi import APIRouter

router = APIRouter(
    prefix="/clearData",
    tags=["Clear Data"]
)

@router.delete("/")
async def clear_data():
    file_to_delete = "../../current.csv"
    if os.path.exists(file_to_delete):
        try:
            os.remove(file_to_delete)
            open(file_to_delete, "w")
            return {"message": "File deleted successfully"}
        except Exception as e:
            return {"error": f"Failed to delete file: {str(e)}"}
    else:
        return {"message": "File does not exist"}