from PIL import Image, ImageDraw


def rotate_square_image(image_path, angle, transparent_background=False):
    # Load the image
    img = Image.open(image_path)

    # The image is square
    size = img.size[0]  # Since the image is square, width = height

    # Rotate the image
    rotated_img = img.rotate(angle, expand=True,
                             fillcolor=(255, 255, 255))  # Fill outside area with white when rotating

    # Create a mask to keep only the circle in the center
    mask = Image.new('L', rotated_img.size, 0)
    draw = ImageDraw.Draw(mask)
    # Calculate the center of the rotated image
    cx, cy = rotated_img.size[0] // 2, rotated_img.size[1] // 2
    # Calculate radius to fit the original size
    radius = size // 2
    draw.ellipse(((cx - radius, cy - radius), (cx + radius, cy + radius)), fill=255)  # Draw a filled circle

    # Apply the mask to the rotated image and convert to RGB to drop alpha channel
    final_img = Image.new('RGB', rotated_img.size, (255, 255, 255))  # Use a white background for the new final image

    final_img.paste(rotated_img.convert("RGB"), mask=mask)  # Convert to RGB to ignore the alpha channel

    # Optional: Crop the image back to the original size if desired
    final_img = final_img.crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))

    return final_img


if __name__ == '__main__':
    result_img = rotate_square_image("cifar10/cifar10_64/train/bird/img6.png", 98)
    result_img.show()
