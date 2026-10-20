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
