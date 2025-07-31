from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
from scripton import show
from scripton.canvas import Canvas

import torch

#: Load the model

variant = 'facebook/detr-resnet-50'
model = DetrForObjectDetection.from_pretrained(variant, revision='no_timm')
processor = DetrImageProcessor.from_pretrained(variant, revision='no_timm')

#: Detect

image = Image.open('images/cats.jpg')
with torch.no_grad():
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs, output_attentions=True)

#: Visualize

def visualize_interactively(
    image: Image.Image,
    # shape: (num_heads, h, w, h, w)
    attention: torch.Tensor
):
    # The attention head index
    head_idx = 0
    # The normalized spatial coordinate
    # (0, 0) is the top-left coordinate, (1, 1) is the bottom-right
    pos = (0, 0)

    def show_attention():
        h, w = attention.shape[-2:]
        x, y = pos
        row = round(y * (h - 1))
        col = round(x * (w - 1))
        show(
            attention[head_idx, :, :, row, col],
            title=f'Attention Head: {head_idx}, Position: ({row}, {col})',
            key='attention'
        )

    def update_attention_position(event):
        nonlocal pos
        pos = event.normalized
        show_attention()

    def update_attention_head(event):
        nonlocal head_idx
        try:
            head_idx = min(int(event.key), attention.shape[0] - 1)
        except ValueError:
            return
        show_attention()

    canvas = Canvas.from_image(image)
    canvas.on_hover = update_attention_position
    canvas.on_key_down = update_attention_head
    canvas.interact()


# The spatial dimensions of [downsampled] feature maps.
# If you change the input image, you may need to update this to match your
# new image's resolution (as shown in the video).
h, w = 36, 25

visualize_interactively(
    image=image,
    attention=outputs.encoder_attentions[-1][0].reshape(-1, h, w, h, w)
)