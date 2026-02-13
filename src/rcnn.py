#!/usr/bin/env python3
"""
Use a local Faster R-CNN model when available; otherwise fall back to the
local YOLOv8 detector. The script prefers a local checkpoint at the
torch hub cache path to avoid network downloads / SSL issues.
"""
import os
import sys
import shutil
import subprocess
import glob
import argparse

BASE = '/Users/cellben/Desktop/CASEF2026/ridesight/src'
VIDEO_PATH = os.path.join(BASE, 'video1.mp4')
OUTPUT_PATH = os.path.join(BASE, 'video1_rcnn.mp4')


def find_local_fasterrcnn_checkpoint():
    """Search common torch cache locations for a Faster R-CNN checkpoint."""
    candidates = []
    # Default torch hub cache
    home = os.path.expanduser('~')
    candidates += glob.glob(os.path.join(home, '.cache', 'torch', 'hub', 'checkpoints', 'fasterrcnn*.pth'))
    # Workspace-level cache
    candidates += glob.glob(os.path.join(BASE, '..', '.cache', 'torch', 'hub', 'checkpoints', 'fasterrcnn*.pth'))
    return candidates[0] if candidates else None


def run_yolo_fallback(conf=0.5, project_name='yolo_replacement'):
    model_path = os.path.join(BASE, 'yolov8n.pt')
    output_dir = os.path.join(BASE, project_name)
    cmd = [
        'yolo', 'detect', 'predict',
        f'model={model_path}', f'source={VIDEO_PATH}', f'conf={conf}',
        f'project={output_dir}', 'name=runs', 'save=True', 'vid_stride=1'
    ]
    print('Running fallback YOLO command:', ' '.join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError('YOLO fallback failed')
    # copy output video if produced
    out_video = os.path.join(output_dir, 'runs', 'video1.mp4')
    if os.path.exists(out_video):
        shutil.copy(out_video, OUTPUT_PATH)
        print(f'✓ Copied YOLO output to {OUTPUT_PATH}')
        return True
    return False


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f'ERROR: video not found: {VIDEO_PATH}')
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Run Faster R-CNN (local) or YOLO fallback on a video')
    parser.add_argument('--short-side', type=int, default=320, help='resize shorter side to this for model input (<=0 to disable)')
    parser.add_argument('--max-frames', type=int, default=0, help='stop after this many frames (0 = all)')
    parser.add_argument('--frame-skip', type=int, default=1, help='process every Nth frame (1 = every frame)')
    args = parser.parse_args()
    # Try to load torchvision Faster R-CNN using a local checkpoint when possible.
    ckpt = find_local_fasterrcnn_checkpoint()
    if ckpt:
        print('Found local Faster R-CNN checkpoint:', ckpt)
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn

            model = fasterrcnn_resnet50_fpn(pretrained=False)
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
            model.eval()

            # Run inference using the PyTorch model
            import cv2
            from torchvision.transforms import functional as F
            print('Running local Faster R-CNN on video...')
            cap = cv2.VideoCapture(VIDEO_PATH)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'Video properties: {w}x{h} at {fps:.2f} FPS')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
            frame_idx = 0
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if args.frame_skip > 1 and (frame_idx % args.frame_skip) != 0:
                        continue
                    if args.max_frames and frame_idx > args.max_frames:
                        break
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize for faster inference on CPU (scale boxes back to original after)
                    orig_h, orig_w = img_rgb.shape[:2]
                    scale = 1.0
                    if args.short_side and args.short_side > 0:
                        short = min(orig_h, orig_w)
                        scale = float(args.short_side) / float(short)
                    if scale != 1.0:
                        new_w = max(1, int(orig_w * scale))
                        new_h = max(1, int(orig_h * scale))
                        img_for_model = cv2.resize(img_rgb, (new_w, new_h))
                    else:
                        img_for_model = img_rgb
                    t = F.to_tensor(img_for_model).unsqueeze(0)
                    outputs = model(t)
                    img_out = frame.copy()
                    if outputs and len(outputs) > 0:
                        out0 = outputs[0]
                        boxes = out0['boxes'].cpu().numpy()
                        scores = out0['scores'].cpu().numpy()
                        labels = out0['labels'].cpu().numpy()
                        mask = scores >= 0.5
                        # boxes are in resized coordinates if we resized; scale them back
                        if scale != 1.0:
                            boxes = boxes / scale
                        for box, score, label in zip(boxes[mask], scores[mask], labels[mask]):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_out, f'{int(label)}:{score:.2f}', (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    out.write(img_out)
                    if frame_idx % 50 == 0:
                        print(f'  Frame {frame_idx}')
            cap.release()
            out.release()
            print('✓ Wrote', OUTPUT_PATH)
            return
        except Exception as e:
            print('Local Faster R-CNN failed:', str(e))

    # If we didn't find a local checkpoint or it failed, attempt to load torchvision weights (may hit SSL).
    try:
        print('Attempting to load torchvision pretrained Faster R-CNN (network may be required)...')
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        # If loading succeeded, run detection similarly to above
        import cv2
        from torchvision.transforms import functional as F
        print('Running downloaded Faster R-CNN on video...')
        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
        frame_idx = 0
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if args.frame_skip > 1 and (frame_idx % args.frame_skip) != 0:
                    continue
                if args.max_frames and frame_idx > args.max_frames:
                    break
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize for faster inference on CPU
                orig_h, orig_w = img_rgb.shape[:2]
                scale = 1.0
                if args.short_side and args.short_side > 0:
                    short = min(orig_h, orig_w)
                    scale = float(args.short_side) / float(short)
                if scale != 1.0:
                    new_w = max(1, int(orig_w * scale))
                    new_h = max(1, int(orig_h * scale))
                    img_for_model = cv2.resize(img_rgb, (new_w, new_h))
                else:
                    img_for_model = img_rgb
                t = F.to_tensor(img_for_model).unsqueeze(0)
                outputs = model(t)
                img_out = frame.copy()
                if outputs and len(outputs) > 0:
                    out0 = outputs[0]
                    boxes = out0['boxes'].cpu().numpy()
                    scores = out0['scores'].cpu().numpy()
                    labels = out0['labels'].cpu().numpy()
                    mask = scores >= 0.5
                    if scale != 1.0:
                        boxes = boxes / scale
                    for box, score, label in zip(boxes[mask], scores[mask], labels[mask]):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(img_out, (x1, y1), 5, (255, 0, 0), -1)
                        cv2.putText(img_out, f'{int(label)}:{score:.2f}', (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                out.write(img_out)
                if frame_idx % 50 == 0:
                    print(f'  Frame {frame_idx}')
        cap.release()
        out.release()
        print('✓ Wrote', OUTPUT_PATH)
        return
    except Exception as e:
        print('Could not load pretrained torchvision model (likely SSL/network):', str(e))

    # Final fallback: run YOLOv8 (local) and copy its output
    try:
        ok = run_yolo_fallback(conf=0.5, project_name='yolo_replacement')
        if not ok:
            raise RuntimeError('YOLO fallback produced no video')
    except Exception as e:
        print('All detectors failed:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()

