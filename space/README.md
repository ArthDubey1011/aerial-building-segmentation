---
title: Aerial Building Segmentation
sdk: gradio
sdk_version: 6.28.0
app_file: space_app.py
pinned: false
short_description: U-Net building segmentation, count, area % and density heatmap from aerial images
---

# Aerial Building Segmentation & Density Mapping

Upload an aerial image (RGB, ~1 m/pixel works best). The app returns a building mask, individual buildings
(watershed), a building-density heatmap, the building count and the built-up area percentage.

- Model: U-Net with an ImageNet-pretrained ResNet34 encoder, trained on the Massachusetts Buildings dataset
  (test IoU 0.695 / Dice 0.820 on full images).
- Analyses **one image at a time**; it does not detect change over time.
- Building counts are approximate: the watershed step can over-split large roofs and under-split row houses.
- Runs on free CPU hardware; "Fast mode" analyses only the centre 1024x1024 crop.

Source code, training details and honest failure analysis: see the GitHub repository (link in the model card).
