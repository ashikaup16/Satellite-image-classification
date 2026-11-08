"""Streamlit dashboard for satellite image classification and benchmarking."""

from hashlib import sha256
from io import BytesIO
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import numpy as np
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from classifier import ModelLoadError, SatelliteClassifier
from config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TILE_SIZE,
    MAX_UPLOAD_BYTES,
    MAX_IMAGE_PIXELS,
    PROJECT_ROOT,
)
from image_processing import (
    class_percentages,
    combine_masks,
    create_tiles,
    decode_uploaded_image,
    compare_landcover_masks,
    class_transition_counts,
    render_classification,
)
from parallel_processing import run_batched, run_sequential
from performance import calculate_speedup, measure_processing
from paste_input import clipboard_image_input, decode_clipboard_payload


def create_comparison_figure(
    original: np.ndarray,
    classified: np.ndarray,
    mask: np.ndarray,
    visible_class_ids: set[int],
    tile_size: int,
    show_tile_grid: bool,
    classified_title: str,
) -> plt.Figure:
    """Create a side-by-side source and labeled segmentation figure."""
    figure, (original_axis, result_axis) = plt.subplots(
        1, 2, figsize=(13, 5.5), constrained_layout=True
    )
    original_axis.imshow(original)
    original_axis.set_title("(a) Original satellite image", fontsize=13, pad=12)
    original_axis.axis("off")

    result_axis.imshow(classified)
    result_axis.set_title(classified_title, fontsize=13, pad=12)
    result_axis.axis("off")

    if show_tile_grid:
        for axis in (original_axis, result_axis):
            axis.set_xticks(np.arange(tile_size, original.shape[1], tile_size), minor=True)
            axis.set_yticks(np.arange(tile_size, original.shape[0], tile_size), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.7, alpha=0.8)
            axis.tick_params(which="minor", bottom=False, left=False)

    offsets = {
        0: (-24, 24),
        1: (-28, -28),
        2: (28, -24),
        3: (28, 26),
    }
    for class_id, (name, offset) in enumerate(zip(CLASS_NAMES, offsets.values())):
        if class_id not in visible_class_ids:
            continue
        component_count, _, statistics, centroids = cv2.connectedComponentsWithStats(
            (mask == class_id).astype(np.uint8), connectivity=8
        )
        if component_count <= 1:
            continue
        largest_component = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
        center_x, center_y = centroids[largest_component]
        result_axis.annotate(
            name,
            xy=(center_x, center_y),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            color="#1b2630",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#d3dbe2",
                "alpha": 0.96,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": "white",
                "linewidth": 1.8,
                "shrinkA": 5,
                "shrinkB": 4,
            },
        )
    return figure


def create_change_figure(
    before_image: np.ndarray,
    after_image: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    changed: np.ndarray,
    water_gain: np.ndarray,
) -> plt.Figure:
    """Render source images, class maps, and highlighted change candidates."""
    palette = np.asarray(CLASS_COLORS, dtype=np.uint8)
    change_colors = after_image.copy()
    flood_colors = after_image.copy()
    change_colors[changed] = (
        0.35 * change_colors[changed] + 0.65 * np.array([255, 190, 45])
    ).astype(np.uint8)
    flood_colors[water_gain] = (
        0.25 * flood_colors[water_gain] + 0.75 * np.array([30, 190, 235])
    ).astype(np.uint8)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = (
        (before_image, "Before · original"),
        (after_image, "After · original"),
        (palette[before], "Before · land-cover classes"),
        (palette[after], "After · land-cover classes"),
        (change_colors, "Changed pixels highlighted"),
        (flood_colors, "New-water candidates highlighted"),
    )
    for axis, (pixels, title) in zip(axes.flat, panels):
        axis.imshow(pixels)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    return figure


def create_pdf_report(
    image: np.ndarray,
    mask: np.ndarray,
    runs: dict,
    tile_count: int,
    model_description: str,
    after_image: np.ndarray | None = None,
    after_mask: np.ndarray | None = None,
    changed: np.ndarray | None = None,
    water_gain: np.ndarray | None = None,
    water_loss: np.ndarray | None = None,
    ground_scale: float | None = None,
) -> bytes:
    """Build a compact PDF summary with classification and optional change maps."""
    output = BytesIO()
    percentages_before = class_percentages(mask)
    pixels_before = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))[:len(CLASS_NAMES)]
    with PdfPages(output) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3))
        figure.suptitle("Satellite Image Classification Report", fontsize=22, y=0.94)
        summary_lines = [
            f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
            f"Image dimensions: {image.shape[1]} × {image.shape[0]} pixels",
            f"Tiles processed: {tile_count}",
            f"Model: {model_description}",
            "",
            "Land-cover coverage (before image):",
        ]
        summary_lines.extend(
            f"  {name}: {percentage:.2f}% ({int(count):,} pixels)"
            for name, percentage, count in zip(
                CLASS_NAMES, percentages_before, pixels_before
            )
        )
        if ground_scale is not None:
            summary_lines.append(
                f"Ground sample distance: {ground_scale:g} meters/pixel "
                "(area estimates assume square, uniformly scaled pixels)"
            )
            summary_lines.extend(
                f"  {name} estimated area: {count * ground_scale**2 / 10_000:.4f} ha"
                for name, count in zip(CLASS_NAMES, pixels_before)
            )
        summary_lines.extend(("", "Processing performance:"))
        for label, timed in runs.items():
            summary_lines.append(
                f"  {label}: {timed.seconds:.4f} seconds "
                f"({timed.result.execution_device})"
            )
        sequential = runs.get("sequential")
        batched = runs.get("batched")
        if sequential and batched:
            speedup = calculate_speedup(sequential.seconds, batched.seconds)
            if speedup is not None:
                summary_lines.append(f"  Sequential / batched speedup: {speedup:.3f}×")
        if after_mask is not None:
            percentages_after = class_percentages(after_mask)
            pixels_after = np.bincount(
                after_mask.ravel(), minlength=len(CLASS_NAMES)
            )[:len(CLASS_NAMES)]
            summary_lines.extend(("", "Land-cover coverage (after image):"))
            summary_lines.extend(
                f"  {name}: {percentage:.2f}% ({int(count):,} pixels)"
                for name, percentage, count in zip(
                    CLASS_NAMES, percentages_after, pixels_after
                )
            )
            summary_lines.extend(("", "Per-class change (after minus before):"))
            summary_lines.extend(
                f"  {name}: {after_percentage - before_percentage:+.2f} "
                f"percentage points ({int(after_count) - int(before_count):+,} pixels)"
                for name, before_percentage, after_percentage, before_count, after_count in zip(
                    CLASS_NAMES,
                    percentages_before,
                    percentages_after,
                    pixels_before,
                    pixels_after,
                )
            )
        if changed is not None and water_gain is not None and water_loss is not None:
            changed_count = int(np.count_nonzero(changed))
            water_gain_count = int(np.count_nonzero(water_gain))
            water_loss_count = int(np.count_nonzero(water_loss))
            total_pixels = changed.size
            summary_lines.extend(
                (
                    "",
                    "Before/after change summary:",
                    f"  Changed land-cover pixels: {changed_count:,} "
                    f"({changed_count / total_pixels * 100:.2f}%)",
                    f"  New water / flood candidates: {water_gain_count:,} "
                    f"({water_gain_count / total_pixels * 100:.2f}%)",
                    f"  Water decrease: {water_loss_count:,} "
                    f"({water_loss_count / total_pixels * 100:.2f}%)",
                )
            )
            if ground_scale is not None:
                summary_lines.append(
                    f"  New water estimated area: "
                    f"{water_gain_count * ground_scale**2 / 10_000:.4f} ha"
                )
        figure.text(
            0.08, 0.86, "\n".join(summary_lines),
            va="top", fontsize=12, linespacing=1.5,
        )
        figure.text(
            0.08, 0.055,
            "Change and flood-candidate maps are pixel-wise comparisons and "
            "require spatially aligned images. They are not a substitute for "
            "validated emergency or scientific assessment.",
            fontsize=9, color="#4d5b68", wrap=True,
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(11.7, 6), constrained_layout=True)
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[0].axis("off")
        axes[1].imshow(np.asarray(CLASS_COLORS, dtype=np.uint8)[mask])
        axes[1].set_title("Four-class land-cover map")
        axes[1].axis("off")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        if (
            after_image is not None
            and after_mask is not None
            and changed is not None
            and water_gain is not None
        ):
            change_figure = create_change_figure(
                image, after_image, mask, after_mask, changed, water_gain
            )
            pdf.savefig(change_figure, bbox_inches="tight")
            plt.close(change_figure)
    return output.getvalue()


st.set_page_config(
    page_title="Satellite Image Classification Accelerator",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
        color-scheme: light;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 8%, rgba(52, 147, 168, .13), transparent 34%),
            radial-gradient(ellipse at 92% 18%, rgba(66, 122, 184, .11), transparent 31%),
            linear-gradient(145deg, #f5f8fc 0%, #eaf1f8 48%, #f5f8fc 100%);
        background-size: 150% 150%;
        animation: ambientShift 24s ease-in-out infinite alternate;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        animation: contentEnter .65s ease-out both;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #e8f0f7 100%);
        border-right: 1px solid rgba(29, 68, 91, .08);
    }
    .hero {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        padding: 1.8rem 2rem;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 22px;
        color: white;
        background: linear-gradient(115deg, #102d46 0%, #14536c 55%, #16808a 100%);
        box-shadow: 0 16px 42px rgba(16, 45, 70, .19);
        margin-bottom: 1.4rem;
        animation: heroEnter .7s ease-out both;
    }
    .hero::after {
        content: "";
        position: absolute;
        z-index: -1;
        top: -80%;
        left: -45%;
        width: 32%;
        height: 260%;
        transform: rotate(25deg);
        background: linear-gradient(90deg, transparent, rgba(255,255,255,.14), transparent);
        animation: heroShimmer 10s ease-in-out 1s infinite;
    }
    .hero h1 {
        letter-spacing: -.035em;
        line-height: 1.16;
        animation: titleEnter .8s .12s both;
    }
    .hero p { color: #d7ecf1; margin-bottom: 0; }
    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(20, 83, 108, .10);
        border-radius: 16px;
        background: rgba(255,255,255,.72);
        box-shadow: 0 8px 24px rgba(29, 68, 91, .06);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(29, 68, 91, .12);
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 11px;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(20, 83, 108, .16);
        filter: brightness(1.03);
    }
    [data-testid="stFileUploader"] {
        border-radius: 16px;
        background: rgba(255,255,255,.52);
    }
    @keyframes ambientShift {
        from { background-position: 0% 0%; }
        to { background-position: 100% 80%; }
    }
    @keyframes contentEnter {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes heroEnter {
        from { opacity: 0; transform: translateY(-8px) scale(.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes titleEnter {
        from { opacity: 0; transform: translateY(7px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes heroShimmer {
        0%, 66%, 100% { left: -45%; }
        82% { left: 120%; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🛰️ Satellite Image Classification Accelerator</h1>'
    "<p>Compare one-tile-at-a-time inference with batched PyTorch inference. "
    "CUDA is selected automatically when available.</p></div>",
    unsafe_allow_html=True,
)

feature_columns = st.columns(4)
feature_summaries = (
    ("🗺️ 4 land-cover classes", "Land · Vegetation · Water · Urban"),
    ("⚡ Parallel processing", "CPU baseline and CUDA batch when available"),
    ("📐 Area calculation", "Percentages, pixel counts, optional ground scale"),
    ("🌊 Change & flood candidates", "Compare aligned before/after images"),
)
for column, (title, description) in zip(feature_columns, feature_summaries):
    column.markdown(
        f'<div style="padding:14px 15px;border-radius:15px;'
        f'background:rgba(255,255,255,.72);border:1px solid rgba(20,83,108,.10);'
        f'box-shadow:0 6px 20px rgba(29,68,91,.05);min-height:82px">'
        f'<strong>{title}</strong><br><span style="font-size:.82rem;color:#526373">'
        f'{description}</span></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Processing settings")
    tile_size = st.select_slider(
        "Tile size",
        options=[64, 128, 256, 384, 512],
        value=DEFAULT_TILE_SIZE,
        help="Larger tiles reduce tile count but use more memory per inference.",
    )
    batch_size = st.select_slider(
        "Batch size",
        options=[1, 2, 4, 8, 16, 32],
        value=DEFAULT_BATCH_SIZE,
        help="The batched path automatically reduces this after a CUDA memory error.",
    )
    run_sequential_option = st.checkbox("Run sequential benchmark", value=True)
    run_batched_option = st.checkbox("Run batched benchmark", value=True)
    st.divider()
    st.subheader("Map display")
    map_style = st.radio(
        "Classification view",
        ("Pure class colors", "Overlay on original"),
        help="Choose a solid segmentation map or blend the class colors with the image.",
    )
    overlay_opacity = st.slider(
        "Class color opacity",
        min_value=10,
        max_value=100,
        value=60,
        format="%d%%",
        disabled=map_style != "Overlay on original",
    )
    visible_class_names = st.multiselect(
        "Visible classes",
        options=list(CLASS_NAMES),
        default=list(CLASS_NAMES),
        help="Hide classes from the map without changing classification statistics.",
    )
    show_tile_grid = st.checkbox(
        "Show processing tile grid",
        value=False,
        help="Overlay the selected tile boundaries to visualize parallel work units.",
    )
    use_ground_scale = st.checkbox(
        "Estimate ground area",
        value=False,
        help="Enable only when the image's ground sample distance is known.",
    )
    ground_scale = (
        st.number_input(
            "Ground sample distance (meters/pixel)",
            min_value=0.001,
            max_value=10000.0,
            value=0.3,
            step=0.1,
            help="Area estimates assume square pixels and a consistent image scale.",
        )
        if use_ground_scale
        else None
    )

st.subheader("Copy and paste images")
st.caption(
    "Copy an image from any app or webpage, click the matching box, then press "
    "Ctrl+V (⌘+V on Mac). Supports common raster images; no file upload is needed. "
    f"Paste limit: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB and "
    f"{MAX_IMAGE_PIXELS:,} pixels per image."
)
before_column, after_column = st.columns(2)
with before_column:
    before_payload = clipboard_image_input(
        "Paste BEFORE image",
        key="before_clipboard_image",
    )
with after_column:
    after_payload = clipboard_image_input(
        "Paste AFTER image (optional)",
        key="after_clipboard_image",
    )

image = None
after_image = None
upload_hash = None
before_input = after_input = None
try:
    before_input = decode_clipboard_payload(before_payload, "Before image")
    after_input = decode_clipboard_payload(after_payload, "After image")
except ValueError as exc:
    st.error(str(exc))

data, before_name = before_input if before_input is not None else (None, None)
after_data, after_name = after_input if after_input is not None else (None, None)

if data is not None:
    if len(data) > MAX_UPLOAD_BYTES:
        st.error(
            f"Before image is too large. Maximum is "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    else:
        try:
            image = decode_uploaded_image(data, before_name)
            st.caption(
                f"Before image validated: {image.shape[1]} × {image.shape[0]} pixels."
            )
        except ValueError as exc:
            st.error(f"Before image: {exc}")

if after_data is not None:
    if len(after_data) > MAX_UPLOAD_BYTES:
        st.error(
            f"After image is too large. Maximum is "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    else:
        try:
            after_image = decode_uploaded_image(after_data, after_name)
            st.caption(
                f"After image validated: "
                f"{after_image.shape[1]} × {after_image.shape[0]} pixels."
            )
        except ValueError as exc:
            st.error(f"After image: {exc}")

images_aligned = (
    image is not None
    and (after_data is None or (
        after_image is not None and after_image.shape[:2] == image.shape[:2]
    ))
)
if image is not None and after_image is not None:
    if images_aligned:
        st.info(
            "Change analysis compares matching pixel coordinates. Make sure both "
            "images are precisely registered to the same location and scale."
        )
    else:
        st.error(
            "Before and after images have different dimensions. Use aligned images "
            "with identical width and height; they are not resized automatically."
        )

if data is not None and image is not None:
    digest = sha256()
    digest.update(data)
    if after_data is not None:
        digest.update(after_data)
    digest.update(
        (
            f"{tile_size}:{batch_size}:{run_sequential_option}:"
            f"{run_batched_option}"
        ).encode("ascii")
    )
    upload_hash = digest.hexdigest()

if image is not None:
    st.subheader("Input preview")
    if after_image is None:
        st.image(image, caption=before_name, use_container_width=True)
    else:
        preview_before, preview_after = st.columns(2)
        preview_before.image(image, caption=f"Before · {before_name}", use_container_width=True)
        preview_after.image(after_image, caption=f"After · {after_name}", use_container_width=True)

    if not run_sequential_option and not run_batched_option:
        st.warning("Select at least one processing method.")
    classify_clicked = st.button(
        "Run land-cover analysis",
        type="primary",
        disabled=(
            not (run_sequential_option or run_batched_option)
            or not images_aligned
        ),
    )
    if classify_clicked:
        method_count = int(run_sequential_option) + int(run_batched_option)
        work_count = method_count + int(after_image is not None)
        progress = st.progress(0, text="Preparing image tiles…")
        try:
            tiles = create_tiles(image, tile_size)
            after_tiles = create_tiles(after_image, tile_size) if after_image is not None else None
            sequential_classifier = None
            if run_sequential_option:
                # When both methods are selected, the sequential path is the CPU baseline.
                sequential_device = "cpu" if run_batched_option else None
                sequential_classifier = SatelliteClassifier(device=sequential_device)
            batched_classifier = (
                SatelliteClassifier() if run_batched_option else None
            )
            runs = {}
            completed_work = 0

            def progress_for(label: str):
                def update(done: int, total: int) -> None:
                    fraction = done / max(total, 1)
                    progress.progress(
                        min(99, int((completed_work + fraction) / work_count * 100)),
                        text=f"{label}: tile {done}/{total}",
                    )

                return update

            if run_sequential_option:
                runs["sequential"] = measure_processing(
                    lambda: run_sequential(
                        sequential_classifier,
                        tiles,
                        progress_for("CPU sequential" if run_batched_option else "Sequential"),
                    )
                )
                completed_work += 1
            if run_batched_option:
                runs["batched"] = measure_processing(
                    lambda: run_batched(
                        batched_classifier,
                        tiles,
                        batch_size,
                        progress_for("Parallel batch"),
                    )
                )
                completed_work += 1

            selected = runs.get("batched", runs.get("sequential"))
            mask = combine_masks(image.shape, tiles, selected.result.masks)
            after_mask = None
            after_timed = None
            after_device = None
            if after_tiles is not None:
                after_classifier = batched_classifier or sequential_classifier
                if run_batched_option:
                    after_timed = measure_processing(
                        lambda: run_batched(
                            after_classifier,
                            after_tiles,
                            batch_size,
                            progress_for("Classifying after image"),
                        )
                    )
                else:
                    after_timed = measure_processing(
                        lambda: run_sequential(
                            after_classifier,
                            after_tiles,
                            progress_for("Classifying after image"),
                        )
                    )
                after_device = after_timed.result.execution_device
                after_mask = combine_masks(
                    after_image.shape, after_tiles, after_timed.result.masks
                )
            progress.progress(100, text="Analysis complete")

            changed = water_gain = water_loss = None
            if after_mask is not None:
                changed, water_gain, water_loss = compare_landcover_masks(mask, after_mask)
            chosen_classifier = batched_classifier or sequential_classifier
            color_map, overlay = render_classification(image, mask)
            st.session_state["classification_result"] = {
                "upload_hash": upload_hash,
                "mask": mask,
                "color_map": color_map,
                "overlay": overlay,
                "runs": runs,
                "mode": chosen_classifier.mode_description,
                "tile_count": len(tiles),
                "batch_size": batch_size,
                "after_mask": after_mask,
                "after_device": after_device,
                "changed": changed,
                "water_gain": water_gain,
                "water_loss": water_loss,
                "after_seconds": after_timed.seconds if after_timed else None,
            }
        except ModelLoadError as exc:
            st.error(str(exc))
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"Analysis could not be completed: {exc}")

result = st.session_state.get("classification_result")
if image is not None and result and result["upload_hash"] == upload_hash:
    st.divider()
    if result["mode"].startswith("Heuristic"):
        st.caption("Classifier: heuristic demonstration mode (no trained model loaded).")
    else:
        st.success(result["mode"])

    st.subheader("Original image and classified regions")
    visible_class_ids = {
        CLASS_NAMES.index(name) for name in visible_class_names
    }
    color_map, overlay = render_classification(
        image,
        result["mask"],
        alpha=overlay_opacity / 100.0,
        visible_classes=visible_class_ids,
    )
    classified_view = overlay if map_style == "Overlay on original" else color_map
    figure_title = (
        "(b) Class overlay" if map_style == "Overlay on original"
        else "(b) Segmentation map"
    )
    comparison_figure = create_comparison_figure(
        image,
        classified_view,
        result["mask"],
        visible_class_ids,
        tile_size,
        show_tile_grid,
        figure_title,
    )
    st.pyplot(comparison_figure, use_container_width=True)
    plt.close(comparison_figure)

    st.markdown("**Legend** · hidden categories appear in charcoal")
    legend_columns = st.columns(len(CLASS_NAMES))
    for index, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
        legend_columns[index].markdown(
            f'<span style="display:inline-block;width:13px;height:13px;'
            f'background:{rgb};border-radius:3px;margin-right:7px"></span>{name}',
            unsafe_allow_html=True,
        )

    st.subheader("Class coverage")
    percentages = class_percentages(result["mask"])
    pixel_counts = np.bincount(
        result["mask"].ravel(), minlength=len(CLASS_NAMES)
    )[:len(CLASS_NAMES)]
    statistic_columns = st.columns(4)
    for column, name, percentage in zip(statistic_columns, CLASS_NAMES, percentages):
        column.metric(name, f"{percentage:.1f}%")

    st.markdown("#### Pixel distribution")
    distribution_figure, distribution_axis = plt.subplots(figsize=(8, 3.2))
    bars = distribution_axis.barh(
        list(CLASS_NAMES),
        percentages,
        color=[tuple(channel / 255 for channel in color) for color in CLASS_COLORS],
    )
    distribution_axis.set_xlabel("Share of classified pixels (%)")
    distribution_axis.set_xlim(0, 100)
    distribution_axis.bar_label(bars, labels=[f"{value:.1f}%" for value in percentages], padding=4)
    distribution_axis.spines[["top", "right"]].set_visible(False)
    distribution_figure.tight_layout()
    st.pyplot(distribution_figure, use_container_width=True)
    plt.close(distribution_figure)

    st.markdown("#### Before / after class-by-class comparison")
    after_mask = result["after_mask"]
    after_percentages = None
    after_pixel_counts = None
    if after_mask is not None:
        after_percentages = class_percentages(after_mask)
        after_pixel_counts = np.bincount(
            after_mask.ravel(), minlength=len(CLASS_NAMES)
        )[:len(CLASS_NAMES)]
    area_rows = []
    for index, name in enumerate(CLASS_NAMES):
        before_area = (
            int(pixel_counts[index]) * ground_scale**2 / 10_000
            if ground_scale is not None else None
        )
        after_area = (
            int(after_pixel_counts[index]) * ground_scale**2 / 10_000
            if ground_scale is not None and after_pixel_counts is not None else None
        )
        row = {
            "Land-cover class": name,
            "Before pixels": int(pixel_counts[index]),
            "Before (%)": round(percentages[index], 2),
            "After pixels": (
                int(after_pixel_counts[index])
                if after_pixel_counts is not None else "—"
            ),
            "After (%)": (
                round(after_percentages[index], 2)
                if after_percentages is not None else "—"
            ),
            "Change (percentage points)": (
                round(after_percentages[index] - percentages[index], 2)
                if after_percentages is not None else "—"
            ),
            "Pixel change": (
                int(after_pixel_counts[index]) - int(pixel_counts[index])
                if after_pixel_counts is not None else "—"
            ),
        }
        if ground_scale is not None:
            row["Before (ha)"] = round(before_area, 4)
            row["After (ha)"] = round(after_area, 4) if after_area is not None else "—"
            row["Area change (ha)"] = (
                round(after_area - before_area, 4) if after_area is not None else "—"
            )
        area_rows.append(row)
    st.table(area_rows)
    if ground_scale is not None:
        st.caption(
            f"Area estimate uses {ground_scale:g} m/pixel and assumes square, "
            "uniformly scaled pixels; it is only meaningful for georeferenced imagery."
        )

    change_figure = None
    change_buffer = BytesIO()
    if after_mask is not None:
        st.subheader("Before/after land-cover change")
        st.caption(
            "Change and flood candidates are based on pixel-wise class transitions. "
            "Images must be precisely co-registered; seasonal, lighting, sensor, and "
            "classification differences can create false changes."
        )
        change_figure = create_change_figure(
            image,
            after_image,
            result["mask"],
            after_mask,
            result["changed"],
            result["water_gain"],
        )
        change_figure.savefig(
            change_buffer, format="png", dpi=160, bbox_inches="tight"
        )
        st.pyplot(change_figure, use_container_width=True)
        plt.close(change_figure)

        total_pixels = result["changed"].size
        changed_count = int(np.count_nonzero(result["changed"]))
        water_gain_count = int(np.count_nonzero(result["water_gain"]))
        water_loss_count = int(np.count_nonzero(result["water_loss"]))
        change_metrics = st.columns(4)
        change_metrics[0].metric(
            "Changed land-cover pixels",
            f"{changed_count / total_pixels * 100:.2f}%",
            f"{changed_count:,} pixels",
        )
        change_metrics[1].metric(
            "New water / flood candidates",
            f"{water_gain_count / total_pixels * 100:.2f}%",
            f"{water_gain_count:,} pixels",
        )
        change_metrics[2].metric(
            "Water decrease",
            f"{water_loss_count / total_pixels * 100:.2f}%",
            f"{water_loss_count:,} pixels",
        )
        if ground_scale is not None:
            water_gain_hectares = water_gain_count * ground_scale**2 / 10_000
            change_metrics[3].metric(
                "New water area estimate",
                f"{water_gain_hectares:.3f} ha",
            )
        else:
            change_metrics[3].metric("After-image inference", f"{result['after_seconds']:.3f} s")
        st.caption(
            f"After image execution device: {result['after_device']}. "
            "A new-water region is a flood candidate only; it is not a validated "
            "flood detection or emergency alert."
        )
        st.markdown("#### Land-cover transition matrix")
        transition_counts = class_transition_counts(result["mask"], after_mask)
        transition_rows = [
            {
                "Before \\ After": before_name,
                **{
                    after_name: int(transition_counts[before_id, after_id])
                    for after_id, after_name in enumerate(CLASS_NAMES)
                },
            }
            for before_id, before_name in enumerate(CLASS_NAMES)
        ]
        st.caption("Cell values are pixel counts; diagonal cells are unchanged pixels.")
        st.table(transition_rows)

        detailed_transitions = [
            {
                "Transition": f"{before_name} → {after_name}",
                "Changed pixels": int(transition_counts[before_id, after_id]),
                "Image share (%)": round(
                    int(transition_counts[before_id, after_id]) / total_pixels * 100,
                    3,
                ),
            }
            for before_id, before_name in enumerate(CLASS_NAMES)
            for after_id, after_name in enumerate(CLASS_NAMES)
            if before_id != after_id and transition_counts[before_id, after_id] > 0
        ]
        if detailed_transitions:
            st.markdown("#### Observed class transitions")
            st.table(
                sorted(
                    detailed_transitions,
                    key=lambda item: item["Changed pixels"],
                    reverse=True,
                )
            )
        else:
            st.success("No class transitions were detected between these inputs.")

    st.subheader("Processing performance")
    metrics = st.columns(4)
    sequential = result["runs"].get("sequential")
    batched = result["runs"].get("batched")
    metrics[0].metric(
        "CPU baseline" if batched else "Sequential time",
        f"{sequential.seconds:.3f} s" if sequential else "Not run",
    )
    metrics[1].metric(
        "Accelerated batch" if batched else "Batched time",
        f"{batched.seconds:.3f} s" if batched else "Not run",
    )
    speedup = (
        calculate_speedup(sequential.seconds, batched.seconds)
        if sequential and batched
        else None
    )
    accelerated_on_cuda = bool(
        batched and batched.result.execution_device == "CUDA"
        and sequential and sequential.result.execution_device == "CPU"
    )
    speedup_label = (
        "CPU vs GPU speedup" if accelerated_on_cuda
        else "CPU sequential/batch" if sequential and batched
        else "Speedup"
    )
    metrics[2].metric(
        speedup_label, f"{speedup:.2f}×" if speedup is not None else "Run both"
    )
    metrics[3].metric("Tiles processed", f"{result['tile_count']:,}")
    for label, run in (("Sequential", sequential), ("Batched", batched)):
        if run:
            display_label = (
                "CPU baseline" if label == "Sequential" and batched
                else "Parallel batch" if label == "Batched"
                else label
            )
            st.caption(f"{display_label} device: {run.result.execution_device}")

    if sequential or batched:
        labels = []
        seconds = []
        if sequential:
            labels.append("Sequential")
            seconds.append(sequential.seconds)
        if batched:
            labels.append("Batched")
            seconds.append(batched.seconds)
        figure, axis = plt.subplots(figsize=(7, 3.2))
        bars = axis.bar(labels, seconds, color=["#efb93f", "#1a8090"][:len(labels)])
        axis.set_ylabel("Wall-clock inference time (seconds)")
        axis.set_title(
            "CPU sequential vs GPU batch" if accelerated_on_cuda
            else "Sequential vs batched tile processing"
        )
        axis.bar_label(bars, fmt="%.3f s", padding=3)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    report_bytes = create_pdf_report(
        image=image,
        mask=result["mask"],
        runs=result["runs"],
        tile_count=result["tile_count"],
        model_description=result["mode"],
        after_image=after_image,
        after_mask=after_mask,
        changed=result["changed"],
        water_gain=result["water_gain"],
        water_loss=result["water_loss"],
        ground_scale=ground_scale,
    )

    st.download_button(
        "Download PDF report",
        data=report_bytes,
        file_name="satellite_analysis_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
