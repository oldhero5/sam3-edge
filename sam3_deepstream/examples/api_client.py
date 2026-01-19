#!/usr/bin/env python3
"""
Example client for SAM3 FastAPI server.

Usage:
    # Basic health check
    python api_client.py --health

    # Segment an image with point prompt
    python api_client.py --image photo.jpg --point 0.5,0.5

    # Segment with bounding box
    python api_client.py --image photo.jpg --box 0.1,0.1,0.9,0.9

    # Save output mask
    python api_client.py --image photo.jpg --point 0.5,0.5 --output mask.png
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import httpx


def check_health(base_url: str) -> dict:
    """Check server health status."""
    response = httpx.get(f"{base_url}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def get_stats(base_url: str) -> dict:
    """Get inference statistics."""
    response = httpx.get(f"{base_url}/stats", timeout=10)
    response.raise_for_status()
    return response.json()


def encode_image(base_url: str, image_path: str) -> dict:
    """Encode an image and cache embeddings on server."""
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        response = httpx.post(
            f"{base_url}/encode",
            files=files,
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def decode_masks(
    base_url: str,
    points: list = None,
    boxes: list = None,
) -> dict:
    """Decode masks from prompts (requires prior encode call)."""
    data = {
        "multimask_output": True,
    }

    if points:
        data["points"] = [
            {"x": p[0], "y": p[1], "label": p[2] if len(p) > 2 else 1}
            for p in points
        ]

    if boxes:
        data["boxes"] = [
            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
            for b in boxes
        ]

    response = httpx.post(
        f"{base_url}/decode",
        json=data,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def segment_image(
    base_url: str,
    image_path: str,
    points: list = None,
    boxes: list = None,
    output_path: str = None,
) -> bytes:
    """One-shot segmentation with optional mask image output."""
    # Build query params
    params = {"return_mask_image": output_path is not None}

    if points:
        params["points"] = ";".join(
            f"{p[0]},{p[1]},{p[2] if len(p) > 2 else 1}" for p in points
        )

    if boxes:
        params["boxes"] = ";".join(
            f"{b[0]},{b[1]},{b[2]},{b[3]}" for b in boxes
        )

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        response = httpx.post(
            f"{base_url}/segment",
            files=files,
            params=params,
            timeout=180,
        )

    response.raise_for_status()

    if output_path and response.headers.get("content-type", "").startswith("image/"):
        # Save mask image
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Mask saved to: {output_path}")

        # Print inference time from headers
        if "X-Inference-Time-Ms" in response.headers:
            print(f"Inference time: {response.headers['X-Inference-Time-Ms']}ms")
        if "X-IoU-Score" in response.headers:
            print(f"IoU score: {response.headers['X-IoU-Score']}")

        return response.content

    return response.json()


def main():
    parser = argparse.ArgumentParser(description="SAM3 API Client")
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="API base URL"
    )
    parser.add_argument(
        "--health", action="store_true",
        help="Check server health"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Get inference statistics"
    )
    parser.add_argument(
        "--image", type=str,
        help="Image file to segment"
    )
    parser.add_argument(
        "--point", type=str, action="append",
        help="Point prompt as 'x,y' or 'x,y,label' (can specify multiple)"
    )
    parser.add_argument(
        "--box", type=str, action="append",
        help="Box prompt as 'x1,y1,x2,y2' (can specify multiple)"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output path for mask image"
    )
    parser.add_argument(
        "--encode-only", action="store_true",
        help="Only encode image (don't decode masks)"
    )

    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    try:
        if args.health:
            result = check_health(base_url)
            print("Health Status:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            return

        if args.stats:
            result = get_stats(base_url)
            print("Inference Statistics:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            return

        if not args.image:
            parser.print_help()
            sys.exit(1)

        # Parse prompts
        points = []
        if args.point:
            for p in args.point:
                parts = [float(x) for x in p.split(",")]
                points.append(parts)

        boxes = []
        if args.box:
            for b in args.box:
                parts = [float(x) for x in b.split(",")]
                boxes.append(parts)

        if args.encode_only:
            # Just encode
            result = encode_image(base_url, args.image)
            print("Encode Result:")
            print(f"  Shape: {result['embedding_shape']}")
            print(f"  Time: {result['inference_time_ms']:.1f}ms")
        elif args.output:
            # Full segmentation with mask output
            segment_image(
                base_url,
                args.image,
                points=points if points else None,
                boxes=boxes if boxes else None,
                output_path=args.output,
            )
        else:
            # Encode then decode
            print("Encoding image...")
            encode_result = encode_image(base_url, args.image)
            print(f"  Encode time: {encode_result['inference_time_ms']:.1f}ms")

            print("Decoding masks...")
            decode_result = decode_masks(
                base_url,
                points=points if points else None,
                boxes=boxes if boxes else None,
            )
            print(f"  Decode time: {decode_result['inference_time_ms']:.1f}ms")
            print(f"  Num masks: {decode_result['num_masks']}")
            print(f"  Mask shape: {decode_result['mask_shape']}")
            print(f"  IoU predictions: {decode_result['iou_predictions']}")

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"  {e.response.text}")
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Could not connect to {base_url}")
        print("Make sure the server is running.")
        sys.exit(1)


if __name__ == "__main__":
    main()
