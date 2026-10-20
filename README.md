# Image Vibe Representation

We can get a vibe from photo!

## Model Structure

```
Image
 ↓
Vision Encoder
 ↓
Low / Mid / High level attributes
 ↓
Vibe representation
```

## Details
### Vision Encoder (Zero-shot)
CLIP, SigLIP, DINOv2

The encoders load their Hugging Face model IDs from `configs/models.yaml`, select
CUDA, MPS, or CPU (in that order), and return L2-normalized embeddings. DINOv2
uses the final hidden state's CLS token as its global image representation.

Install the local inference dependencies:

```bash
pip install -r requirements.txt
```

Extract an image embedding (the model is downloaded by Transformers on first use):

```bash
python scripts/infer.py --encoder clip --image data/sample/example.jpg
python scripts/infer.py --encoder siglip --image data/sample/example.jpg
python scripts/infer.py --encoder dinov2 --image data/sample/example.jpg
```

Use `--device cpu` (or `cuda`, `cuda:0`, `mps`) to override automatic device
selection. CLIP and SigLIP also expose `encode_text(list[str])`; each returned
row is L2-normalized.

### Hierarchical Vibe Structure
```
Low-level
- lighting
- color
- camera
- texture

Mid-level
- composition
- scene
- styling

High-level
- mood
- aesthetic
```
