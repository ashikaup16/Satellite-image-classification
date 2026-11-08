# 🛰️ Satellite Image Classification Accelerator

> 🚀 **Live app:** [Open Satellite Image Classification Accelerator](https://satellite-image-classifiy.streamlit.app/)

> **A Streamlit + PyTorch project for exploring four-class land-cover mapping, image comparison, and sequential-versus-batched inference.**

Paste a satellite image into the dashboard to generate a color-coded land-cover map. Optionally paste a second, aligned image to compare predicted land-cover changes. The project includes a CPU-compatible inference pipeline, optional CUDA acceleration, performance metrics, and a downloadable PDF report.

> [!IMPORTANT]
> **Accuracy notice:** No trained model is included in this repository. By default, the app uses a clearly labeled RGB/texture heuristic demonstration—not a trained semantic segmentation model. Results can be incorrect and are not suitable for operational, scientific, or emergency decisions. Train and validate a model on representative, labeled satellite imagery before relying on its predictions.

## ✨ Features

| Feature | What it does |
|---|---|
| 🖼️ Clipboard image input | Paste a before image and, optionally, an after image directly into the dashboard. |
| 🗺️ Four land-cover classes | Produces a pixel-wise map for **Land**, **Vegetation**, **Water**, and **Urban**. |
| 🎨 Classification visualization | Displays the source image and colored map; switch between pure class colors and an overlay, adjust opacity, filter visible classes, and show tile boundaries. |
| 🧩 Image tiling | Splits large images into configurable square tiles, pads edge tiles for inference, then stitches predictions back together. |
| ⚡ Sequential and batched inference | Compares one-tile-at-a-time processing with batches of independent tiles using the same model and image. |
| 🚀 Automatic device selection | Uses CUDA for batched inference when available; otherwise clearly labels the CPU fallback. |
| 📊 Performance analysis | Reports inference times, execution devices, tile count, and sequential-to-batched speedup. |
| 📈 Class coverage | Shows each class’s predicted pixel share and distribution. |
| 🔎 Before/after comparison | Shows both images and maps, per-class changes, changed-pixel totals, and a class-transition matrix. |
| 🌊 Water-change candidates | Highlights pixels newly predicted as water and pixels no longer predicted as water. These are candidates, not verified flood detections. |
| 📐 Optional area estimates | Converts pixel counts to estimated area when a ground sample distance is provided. |
| 📄 PDF report | Creates a downloadable PDF containing the analysis and its visual results. **The PDF is the only download option.** |
| 🛡️ Input safeguards | Validates pasted raster image data, supported formats, payload size, and pixel dimensions. |

### Class colors

| Class | Map color | Class ID for training masks |
|---|---|---:|
| Land | 🟨 Yellow | `0` |
| Vegetation | 🟩 Green | `1` |
| Water | 🟦 Blue | `2` |
| Urban | 🟥 Red | `3` |

## 🧰 Technology stack and skills demonstrated

- **Python** — modular application code, validation, timing, and training workflow.
- **Streamlit** — interactive dashboard, image-paste component, analysis controls, metrics, tables, and report download.
- **PyTorch** — tensor processing, inference, model loading, training, and CPU/GPU device management.
- **CUDA** — optional NVIDIA GPU inference, selected automatically when supported by the installed PyTorch environment.
- **TorchScript** — portable loading interface for the trained segmentation model.
- **Torchvision** — DeepLabV3-ResNet50 architecture used by the training script.
- **NumPy** — image arrays, masks, class statistics, and transition counts.
- **OpenCV** — image compositing and visualization helpers.
- **Pillow** — raster image decoding, validation, and training dataset loading.
- **Matplotlib** — charts and figures embedded in the dashboard and PDF report.
- **Computer vision** — image tiling, segmentation maps, class coverage, and pixel-wise change analysis.
- **Parallel computing concepts** — independent tile batches, CPU sequential baselines, CUDA batching, timing, and speedup analysis.
- **Testing** — unit tests for decoding and validation, tiling, class maps, change analysis, and sequential/batched output consistency.

## 🚀 Run locally in VS Code

### 1. Prerequisites

- Python **3.10 or later**.
- VS Code with the Python extension.
- For GPU acceleration: a compatible NVIDIA GPU, driver, and CUDA-enabled PyTorch build. CPU-only use is supported.

### 2. Create and activate a virtual environment

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the PyTorch and torchvision versions appropriate for your platform and CUDA setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). For CPU-only inference, select the CPU build. Then install the remaining project dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 4. Launch the dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>. In VS Code, select the `.venv` interpreter if prompted.

### 5. Analyze an image

1. Copy an image from an image viewer or webpage.
2. Click the **Paste BEFORE image** box and press **Ctrl+V** (or **⌘+V** on macOS).
3. Optionally paste a second image into **Paste AFTER image** for comparison.
4. Choose tile size, batch size, display options, and which benchmarks to run in the sidebar.
5. Select **Run land-cover analysis**.
6. Review the class maps, percentages, comparison and performance metrics; use **Download PDF report** to save the report.

The clipboard component accepts common raster formats: JPEG, PNG, WebP, BMP, TIFF, and GIF (first frame). PDF and SVG are vector formats and must be copied/rendered as an image first. Each pasted image is limited to **50 MB** and **50 million pixels**; literally unlimited image sizes are not practical for a browser application.

## ☁️ Deploy on Streamlit Community Cloud

This project is a **Streamlit server application**. Deploy it on
[Streamlit Community Cloud](https://share.streamlit.io/), which runs Streamlit
apps directly from GitHub. Do not select Vercel for this repository: Vercel's
serverless deployment model does not host this Streamlit server and its
interactive browser connection as-is. Adding a Vercel configuration file alone
will not make the app work.

1. Push the project to a GitHub repository that your Streamlit account can
   access. The current repository is
   [`ashikaup16/Satellite-image-classification`](https://github.com/ashikaup16/Satellite-image-classification),
   on the `main` branch.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and
   choose **Create app**.
3. Select the repository `ashikaup16/Satellite-image-classification`, branch
   `main`, and main file path `app.py`.
4. Choose **Deploy** and wait for the dependency installation and first app
   startup to finish. Check the app's build logs in Streamlit Community Cloud if
   startup fails.

The root `requirements.txt` contains the Python dependencies and
`.streamlit/config.toml` contains Streamlit server settings; no Vercel
configuration or secrets are required. The deployed app runs on CPU, so the
batched inference metric will be labeled as a CPU fallback and will **not**
represent GPU acceleration. Streamlit Community Cloud resource limits apply;
very large pasted images may exceed available memory or processing time.

The Vercel screenshot's “project already exists” message means a project with
that name is already in your Vercel account; creating it again is unnecessary.
That project will still not serve this Streamlit app. You can leave the Vercel
project unused and deploy this repository using the steps above.

## 📊 How to interpret the metrics

### Class percentage

For each class:

```text
class coverage (%) = pixels predicted as that class / total image pixels × 100
```

The four class percentages sum to approximately 100%. They describe **model predictions**, not independently verified ground truth.

### Sequential-versus-batched speedup

```text
speedup = sequential inference time / batched inference time
```

- A value **greater than 1×** means batching took less time in that run.
- **1×** means both measured times were equal.
- A value **below 1×** means batching was slower for that run.

When both benchmarks are selected, the sequential path is the CPU baseline; the batched path uses CUDA when available and otherwise uses a labeled CPU batch fallback. Therefore, a CPU-only run measures **CPU sequential versus CPU batched** performance, not GPU speedup. Small images, batch size, device transfers, startup, and system load can all affect timing. A speedup is a runtime measurement, **not an accuracy score**.

CUDA operations are synchronized around timing so that the measurement includes completed GPU work. If GPU memory runs out, batch size is reduced; if necessary, inference falls back to CPU and reports that device.

### Before/after and water candidates

Before/after analysis compares corresponding pixel locations in the two predicted masks. The images must be the same size **and accurately co-registered** (the same ground locations at matching pixel coordinates). The app does not geometrically align images. Differences in alignment, season, lighting, sensor, or classification can look like land-cover changes.

“New water” means a pixel was predicted as non-water before and water after. It is a **water-change candidate**, not validated flood detection, an emergency alert, or proof that flooding occurred.

### Area estimate

If you provide ground sample distance in meters per pixel:

```text
estimated area (hectares) = pixel count × (meters per pixel)² / 10,000
```

This assumes square, uniformly scaled pixels. Use it only with an appropriate, georeferenced image and a valid ground scale. Otherwise, interpret the pixel counts and percentages only.

## 🧠 Train and install a segmentation model

The training script expects paired image and single-channel mask files with matching filename stems:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

Masks must contain only the class IDs `0` (Land), `1` (Vegetation), `2` (Water), and `3` (Urban). Use correctly labeled, representative images and masks; keep the label IDs consistent across all splits.

Run training from the project root:

```powershell
python train.py --dataset dataset --epochs 30 --batch-size 8 --image-size 256
```

The script trains a four-class DeepLabV3-ResNet50 model, reports training loss and validation pixel accuracy per epoch, and writes a TorchScript model to:

```text
models/classifier.pt
```

The exported model must accept a float RGB tensor of shape `[batch, 3, height, width]` with values in `[0, 1]` and return four-class logits of shape `[batch, 4, height, width]`. The example training script starts without pretrained weights. Training alone does not guarantee useful predictions: evaluate on a representative held-out dataset and appropriate metrics before using a checkpoint.

If `models/classifier.pt` is absent, the application explicitly uses heuristic demonstration mode. Its RGB/texture rules are not learned and cannot reliably distinguish all land-cover types, sensors, seasons, or lighting conditions.

## 🧪 Run tests

From the project root:

```powershell
python -B -m unittest discover -s tests -v
```

## 📁 Project structure

```text
.
├── app.py                         # Streamlit UI, analysis flow, charts, PDF report
├── classifier.py                  # TorchScript loading and demo classifier
├── image_processing.py            # Image validation, tiling, masks, comparisons
├── parallel_processing.py         # Sequential and batched tile inference
├── performance.py                 # Synchronized timing and speedup calculation
├── config.py                      # Shared classes, colors, and size limits
├── paste_input.py                 # Clipboard component and payload validation
├── components/
│   └── clipboard_image/           # Browser clipboard image-paste UI
├── train.py                       # DeepLabV3 training and TorchScript export
├── tests/
│   └── test_pipeline.py            # Pipeline and validation tests
├── requirements.txt
├── models/                         # Place classifier.pt here after training
├── uploads/                        # Reserved; pasted images are not retained
├── outputs/                        # Reserved for future persistence
└── utils/
```

## 🔐 Image handling

Pasted image bytes are processed in memory for the current analysis. The application does not save pasted images to `uploads/` or analysis outputs to `outputs/`. Those folders are included as placeholders for future extensions.

## 📄 License

No license is currently specified. Add a license file before redistributing this project or accepting external contributions.
