import os
from PIL import Image, ImageEnhance
import numpy as np

def fix_textures():
    # 1. Fix Gravity Map
    print("Fixing Gravity Map...")
    grav_img = Image.open('../public/textures/gravity_moon_real.jpg').convert('RGB')
    # Force 2:1 aspect ratio exactly to remove the seam
    grav_img = grav_img.resize((1600, 800), Image.Resampling.LANCZOS)
    grav_img.save('../public/textures/gravity_moon_real.jpg', quality=95)
    
    # 2. Fix Aesthetic Map (Galileo False Color)
    print("Enhancing Aesthetic Map to Galileo False Color...")
    base_img = Image.open('../public/textures/8k_moon.jpg').convert('RGB')
    # Resize to 4K to speed up processing
    base_img = base_img.resize((4096, 2048), Image.Resampling.LANCZOS)
    
    # Galileo False Color is created by wildly exaggerating the saturation
    # and adjusting color balance.
    # Convert to numpy for fast vectorized processing
    arr = np.array(base_img, dtype=np.float32) / 255.0
    
    # Extract channels
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    
    # Calculate luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Increase saturation drastically (e.g. 15x) but keep luminance
    sat_factor = 12.0
    
    r_new = lum + (r - lum) * sat_factor
    g_new = lum + (g - lum) * sat_factor
    b_new = lum + (b - lum) * sat_factor
    
    # Push the blue/red differences further:
    # High titanium = more blue. High iron = more orange/red.
    r_new = r_new * 1.1  # slightly boost reds
    b_new = b_new * 1.2  # boost blues more to make titanium maria pop
    
    # Recombine and clip
    arr_new = np.stack((r_new, g_new, b_new), axis=2)
    arr_new = np.clip(arr_new, 0, 1)
    
    out_img = Image.fromarray((arr_new * 255).astype(np.uint8))
    
    # Final contrast and sharpness boost
    out_img = ImageEnhance.Contrast(out_img).enhance(1.2)
    out_img = ImageEnhance.Brightness(out_img).enhance(1.0)
    
    print("Saving aesthetic_moon_real.jpg...")
    out_img.save('../public/textures/aesthetic_moon_real.jpg', quality=95)
    print("Done!")

if __name__ == '__main__':
    fix_textures()
