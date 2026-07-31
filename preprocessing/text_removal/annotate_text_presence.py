"""
Web-based annotation tool for labeling whether manga patches contain text or not.

This tool:
1. Displays candidate text-free crops in a web browser
2. Allows navigation between images using left/right arrow keys
3. Allows changing binary labels (text present or not) using up/down arrow keys
4. Auto-saves annotations to a JSON file

Controls:
- Left Arrow: Previous image
- Right Arrow: Next image
- Up Arrow: Mark as "has text" (label = 1)
- Down Arrow: Mark as "no text" (label = 0)
- S: Save annotations manually
- J: Jump to specific image index
- U: Jump to first unlabeled image
"""

import os
import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import base64
import threading
import webbrowser


def collect_all_images(dataset_dir: str) -> List[str]:
    """
    Collect all PNG images from all subdirectories.

    Args:
        dataset_dir: Root directory containing book subdirectories

    Returns:
        Sorted list of paths to all images
    """
    dataset_path = Path(dataset_dir)
    all_images = []

    # Iterate through all subdirectories (book folders)
    for book_dir in sorted(dataset_path.iterdir()):
        if book_dir.is_dir():
            # Collect all PNG files in this book folder
            for img_file in sorted(book_dir.glob('*.png')):
                all_images.append(str(img_file))

    return all_images


def load_annotations(annotation_file: str) -> Dict:
    """
    Load existing annotations from JSON file.
    """
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'annotations': {},
        'metadata': {
            'created': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'description': 'Binary labels for text presence in manga patches (1=has_text, 0=no_text)'
        }
    }


def save_annotations(annotations: Dict, annotation_file: str) -> None:
    """
    Save annotations to JSON file.
    """
    annotations['metadata']['last_modified'] = datetime.now().isoformat()
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def get_relative_path(image_path: str, dataset_dir: str) -> str:
    """
    Get relative path from dataset directory for use as key.
    """
    return os.path.relpath(image_path, dataset_dir)


def find_first_unlabeled(annotations: Dict, image_paths: List[str], dataset_dir: str) -> int:
    """
    Find the index of the first unlabeled image.
    """
    for i, img_path in enumerate(image_paths):
        rel_path = get_relative_path(img_path, dataset_dir)
        if rel_path not in annotations['annotations']:
            return i
    return 0


def count_labeled(annotations: Dict) -> int:
    """Count number of labeled images."""
    return len(annotations['annotations'])


# Global state
class AppState:
    def __init__(self):
        self.image_paths = []
        self.annotations = {}
        self.annotation_file = ""
        self.dataset_dir = ""
        self.current_idx = 0

state = AppState()


def relative_path(value: str) -> str:
    """Reject machine-specific input and output paths."""
    if Path(value).is_absolute():
        raise argparse.ArgumentTypeError("paths must be relative")
    return value


def create_html_page():
    """Generate the HTML page for annotation."""
    if not state.image_paths:
        return "<html><body><h1>No images found</h1></body></html>"

    img_path = state.image_paths[state.current_idx]
    rel_path = get_relative_path(img_path, state.dataset_dir)

    # Read and encode image as base64
    with open(img_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')

    # Get current label
    current_label = state.annotations['annotations'].get(rel_path, None)

    if current_label is None:
        label_text = "UNLABELED"
        label_color = "#808080"
        label_fg = "black"
    elif current_label == 1:
        label_text = "HAS TEXT (1)"
        label_color = "#00cc00"
        label_fg = "black"
    else:
        label_text = "NO TEXT (0)"
        label_color = "#cc0000"
        label_fg = "white"

    total = len(state.image_paths)
    labeled = count_labeled(state.annotations)
    has_text_count = sum(1 for v in state.annotations['annotations'].values() if v == 1)
    no_text_count = sum(1 for v in state.annotations['annotations'].values() if v == 0)
    unlabeled_count = total - labeled

    has_text_pct = 100 * has_text_count / total if total > 0 else 0
    no_text_pct = 100 * no_text_count / total if total > 0 else 0
    unlabeled_pct = 100 * unlabeled_count / total if total > 0 else 0

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Manga Text Annotation Tool</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a1a;
            color: white;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .main-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 10px;
            overflow: hidden;
        }}
        .image-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            background: #282828;
            border-radius: 8px;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .status-bar {{
            background: #333;
            padding: 15px 20px;
            display: flex;
            align-items: center;
            gap: 20px;
            border-radius: 8px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        .label-indicator {{
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 16px;
            min-width: 150px;
            text-align: center;
        }}
        .info {{
            flex: 1;
        }}
        .info div {{
            margin: 3px 0;
        }}
        .progress {{
            font-size: 14px;
        }}
        .labeled {{
            font-size: 13px;
            color: #aaa;
        }}
        .stats {{
            font-size: 12px;
            color: #999;
            display: flex;
            gap: 15px;
            margin-top: 5px;
        }}
        .stat-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .stat-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        .stat-has-text {{ background: #00cc00; }}
        .stat-no-text {{ background: #cc0000; }}
        .stat-unlabeled {{ background: #808080; }}
        .filename {{
            font-size: 11px;
            color: #888;
            word-break: break-all;
        }}
        .controls {{
            font-size: 12px;
            color: #666;
            text-align: right;
        }}
        .button-row {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }}
        button {{
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: transform 0.1s;
        }}
        button:hover {{
            transform: scale(1.05);
        }}
        button:active {{
            transform: scale(0.95);
        }}
        .btn-prev, .btn-next {{
            background: #555;
            color: white;
        }}
        .btn-has-text {{
            background: #00aa00;
            color: white;
        }}
        .btn-no-text {{
            background: #aa0000;
            color: white;
        }}
        .btn-save {{
            background: #0066cc;
            color: white;
        }}
        .btn-jump {{
            background: #666;
            color: white;
        }}
        .shortcut {{
            font-size: 10px;
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="image-container">
            <img src="data:image/png;base64,{img_data}" alt="Manga patch">
        </div>
        <div class="status-bar">
            <div class="label-indicator" style="background: {label_color}; color: {label_fg};">
                {label_text}
            </div>
            <div class="info">
                <div class="progress">Progress: {state.current_idx + 1} / {total}</div>
                <div class="stats">
                    <span class="stat-item"><span class="stat-dot stat-has-text"></span>Has Text: {has_text_count} ({has_text_pct:.1f}%)</span>
                    <span class="stat-item"><span class="stat-dot stat-no-text"></span>No Text: {no_text_count} ({no_text_pct:.1f}%)</span>
                    <span class="stat-item"><span class="stat-dot stat-unlabeled"></span>Unlabeled: {unlabeled_count} ({unlabeled_pct:.1f}%)</span>
                </div>
                <div class="filename">{rel_path}</div>
            </div>
            <div class="controls">
                <div class="button-row">
                    <button class="btn-prev" onclick="navigate('prev')">← Prev <span class="shortcut">[←]</span></button>
                    <button class="btn-has-text" onclick="setLabel(1)">Has Text <span class="shortcut">[↑]</span></button>
                    <button class="btn-no-text" onclick="setLabel(0)">No Text <span class="shortcut">[↓]</span></button>
                    <button class="btn-next" onclick="navigate('next')">Next → <span class="shortcut">[→]</span></button>
                </div>
                <div class="button-row">
                    <button class="btn-save" onclick="save()">Save <span class="shortcut">[S]</span></button>
                    <button class="btn-jump" onclick="jump()">Jump <span class="shortcut">[J]</span></button>
                    <button class="btn-jump" onclick="jumpUnlabeled()">Unlabeled <span class="shortcut">[U]</span></button>
                </div>
            </div>
        </div>
    </div>
    <script>
        function navigate(dir) {{
            window.location.href = '/action?type=navigate&dir=' + dir;
        }}
        function setLabel(label) {{
            window.location.href = '/action?type=label&value=' + label;
        }}
        function save() {{
            window.location.href = '/action?type=save';
        }}
        function jump() {{
            var idx = prompt('Enter image index (0-{total - 1}):');
            if (idx !== null && idx !== '') {{
                window.location.href = '/action?type=jump&idx=' + idx;
            }}
        }}
        function jumpUnlabeled() {{
            window.location.href = '/action?type=unlabeled';
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.target.tagName === 'INPUT') return;

            switch(e.key) {{
                case 'ArrowLeft':
                    navigate('prev');
                    break;
                case 'ArrowRight':
                    navigate('next');
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    setLabel(1);
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    setLabel(0);
                    break;
                case 's':
                case 'S':
                    save();
                    break;
                case 'j':
                case 'J':
                    jump();
                    break;
                case 'u':
                case 'U':
                    jumpUnlabeled();
                    break;
            }}
        }});
    </script>
</body>
</html>'''
    return html


class AnnotationHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            # Serve main page
            html = create_html_page()
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        elif parsed.path == '/action':
            # Handle actions
            params = urllib.parse.parse_qs(parsed.query)
            action_type = params.get('type', [''])[0]

            if action_type == 'navigate':
                direction = params.get('dir', ['next'])[0]
                if direction == 'prev':
                    state.current_idx = (state.current_idx - 1) % len(state.image_paths)
                else:
                    state.current_idx = (state.current_idx + 1) % len(state.image_paths)

            elif action_type == 'label':
                value = int(params.get('value', [0])[0])
                rel_path = get_relative_path(state.image_paths[state.current_idx], state.dataset_dir)
                state.annotations['annotations'][rel_path] = value
                label_name = "HAS TEXT" if value == 1 else "NO TEXT"
                print(f"Labeled {rel_path}: {label_name} ({value})")
                # Auto-advance
                state.current_idx = (state.current_idx + 1) % len(state.image_paths)

            elif action_type == 'save':
                save_annotations(state.annotations, state.annotation_file)
                labeled = count_labeled(state.annotations)
                print(f"Saved {labeled} annotations to {state.annotation_file}")

            elif action_type == 'jump':
                try:
                    idx = int(params.get('idx', [0])[0])
                    if 0 <= idx < len(state.image_paths):
                        state.current_idx = idx
                except ValueError:
                    pass

            elif action_type == 'unlabeled':
                first_unlabeled = find_first_unlabeled(
                    state.annotations, state.image_paths, state.dataset_dir
                )
                rel_path = get_relative_path(state.image_paths[first_unlabeled], state.dataset_dir)
                if rel_path in state.annotations['annotations']:
                    print("All images are labeled!")
                else:
                    state.current_idx = first_unlabeled
                    print(f"Jumped to first unlabeled image at index {state.current_idx}")

            # Redirect back to main page
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()

        else:
            self.send_error(404)


def run_server(port: int):
    """Run the HTTP server."""
    server = HTTPServer(('0.0.0.0', port), AnnotationHandler)
    print(f"\nServer running at http://localhost:{port}")
    print(f"Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Save before exit
        save_annotations(state.annotations, state.annotation_file)
        labeled = count_labeled(state.annotations)
        print(f"Final save: {labeled} annotations to {state.annotation_file}")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Web-based annotation tool for labeling text presence in manga patches'
    )
    parser.add_argument(
        '--dataset-dir',
        type=relative_path,
        default='data/work/text_removal/candidates',
        help='Directory containing manga patch images'
    )
    parser.add_argument(
        '--annotation-file',
        type=relative_path,
        default='data/work/text_removal/text_presence_annotations.json',
        help='Path to save/load annotations JSON file'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Starting image index (default: 0)'
    )
    parser.add_argument(
        '--start-unlabeled',
        action='store_true',
        help='Start from first unlabeled image'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port for web server (default: 8080)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show annotation statistics and exit'
    )

    args = parser.parse_args()

    # Show stats mode
    if args.stats:
        annotations = load_annotations(args.annotation_file)
        total_labeled = count_labeled(annotations)
        has_text = sum(1 for v in annotations['annotations'].values() if v == 1)
        no_text = sum(1 for v in annotations['annotations'].values() if v == 0)

        image_paths = collect_all_images(args.dataset_dir)
        total_images = len(image_paths)

        print(f"Annotation Statistics:")
        print(f"  Total images: {total_images}")
        print(f"  Labeled: {total_labeled} ({100*total_labeled/total_images:.1f}%)")
        print(f"  Unlabeled: {total_images - total_labeled}")
        print(f"  Has text (1): {has_text}")
        print(f"  No text (0): {no_text}")
        if total_labeled > 0:
            print(f"  Text ratio: {100*has_text/total_labeled:.1f}%")
        return

    # Initialize state
    state.dataset_dir = args.dataset_dir
    state.annotation_file = args.annotation_file

    print("=" * 60)
    print("Manga Text Presence Annotation Tool (Web-based)")
    print("=" * 60)

    # Collect images
    print(f"\nScanning images in {args.dataset_dir}...")
    state.image_paths = collect_all_images(args.dataset_dir)

    if not state.image_paths:
        print("No images found!")
        return

    print(f"Found {len(state.image_paths)} images")

    # Load annotations
    state.annotations = load_annotations(args.annotation_file)
    print(f"Loaded {count_labeled(state.annotations)} existing annotations")

    # Determine start index
    state.current_idx = args.start_index
    if args.start_unlabeled:
        state.current_idx = find_first_unlabeled(
            state.annotations, state.image_paths, state.dataset_dir
        )
        print(f"Starting from first unlabeled image at index {state.current_idx}")

    print()
    print("Controls (in browser):")
    print("  Left/Right Arrow : Navigate between images")
    print("  Up Arrow         : Mark as 'Has Text' (1)")
    print("  Down Arrow       : Mark as 'No Text' (0)")
    print("  S                : Save annotations")
    print("  J                : Jump to specific index")
    print("  U                : Jump to first unlabeled image")
    print()
    print("=" * 60)

    # Run server
    run_server(args.port)


if __name__ == '__main__':
    main()
