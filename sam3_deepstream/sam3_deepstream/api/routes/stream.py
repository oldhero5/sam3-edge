"""WebSocket streaming endpoints."""

import asyncio
import base64
import json
import logging
import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from ..models.responses import RLEMask, StreamResultMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["streaming"])


@router.websocket("/segment")
async def websocket_segment(
    websocket: WebSocket,
    request: Request,
):
    """
    WebSocket endpoint for real-time frame segmentation.

    Protocol:
    - Client sends: {"type": "frame", "data": "<base64>", "frame_idx": N, "prompts": {...}}
    - Server responds: {"type": "result", "frame_idx": N, "masks": [...], "latency_ms": X}

    Connection remains open for streaming multiple frames.
    """
    await websocket.accept()

    # Get processor from app state
    processor = getattr(request.app.state, "frame_processor", None)
    if processor is None:
        await websocket.send_json({
            "type": "error",
            "message": "Frame processor not initialized",
        })
        await websocket.close()
        return

    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive frame data
            message = await websocket.receive_json()
            msg_type = message.get("type", "frame")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "close":
                break

            if msg_type == "frame":
                start_time = time.perf_counter()

                # Decode frame
                frame_data = message.get("data", "")
                frame_idx = message.get("frame_idx", 0)
                prompts = message.get("prompts")

                try:
                    frame = decode_frame(frame_data)
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to decode frame: {str(e)}",
                    })
                    continue

                # Process frame
                try:
                    result = await asyncio.to_thread(
                        processor.process_frame,
                        frame,
                        prompts,
                    )
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Processing error: {str(e)}",
                    })
                    continue

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Encode masks as RLE
                masks = []
                for i, (mask, box, score, obj_id) in enumerate(
                    zip(result.masks, result.boxes, result.scores, result.object_ids)
                ):
                    rle = encode_rle(mask)
                    masks.append(
                        RLEMask(
                            object_id=obj_id,
                            counts=rle["counts"],
                            size=tuple(rle["size"]),
                            score=score,
                            box=box,
                        ).model_dump()
                    )

                # Send result
                response = StreamResultMessage(
                    type="result",
                    frame_idx=frame_idx,
                    latency_ms=latency_ms,
                    masks=masks,
                )
                await websocket.send_json(response.model_dump())

            elif msg_type == "reset":
                # Reset processor state
                processor.reset()
                await websocket.send_json({
                    "type": "reset_complete",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


def decode_frame(data: str) -> np.ndarray:
    """Decode base64 frame data to numpy array."""
    import cv2

    # Decode base64
    image_bytes = base64.b64decode(data)

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Failed to decode image")

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def encode_rle(mask: np.ndarray) -> dict:
    """Encode binary mask as RLE."""
    from ..utils.mask_utils import encode_rle as _encode_rle
    return _encode_rle(mask)
