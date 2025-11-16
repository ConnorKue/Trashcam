from fastapi import APIRouter, HTTPException
import socket
import json

from ..config import settings

router = APIRouter(
    prefix="/fill",
    tags=["Fill Level Check"],
)

@router.get("/")
async def get_fill_level():
    host = settings.UDP_SERVER_HOST
    port = settings.UDP_SERVER_PORT

    if not host:
        raise HTTPException(
            status_code=500,
            detail="UDP_SERVER_HOST is not configured (check your .env)",
        )

    try:
        port = int(port)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=500,
            detail=f"UDP_SERVER_PORT is invalid: {port!r}",
        )

    print(f"[fill] Using UDP target {host}:{port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)

    try:
        try:
            sock.sendto(b"SUBSCRIBE", (host, port))
        except OSError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to send UDP SUBSCRIBE to {host}:{port} - {e}",
            )

        max_packets = 10
        for _ in range(max_packets):
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                raise HTTPException(
                    status_code=504,
                    detail="Timeout waiting for fill-level packet from UDP server",
                )
            except OSError as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"UDP receive failed: {e}",
                )

            text = data.decode("utf-8", errors="ignore").strip()
            if not text.startswith("{"):
                print(f"[fill] Non-JSON packet from {addr}: {text!r}")
                continue

            try:
                depth_info = json.loads(text)
                print(f"[fill] Got depth JSON from {addr}: {depth_info}")
            except json.JSONDecodeError:
                print(f"[fill] Bad JSON packet from {addr}: {text!r}")
                continue

            # ⬇️ THIS is the key line: use fill_percentage
            raw_fill = depth_info.get("fill_percentage")
            if raw_fill is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"UDP JSON missing fill_percentage: {depth_info}",
                )

            try:
                fill = float(raw_fill)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=500,
                    detail=f"fill_percentage is not numeric: {raw_fill!r}",
                )

            # Return only the percentage value
            return {"fillPercent": fill}

        raise HTTPException(
            status_code=502,
            detail="No valid JSON depth packet received from UDP server",
        )

    finally:
        sock.close()
