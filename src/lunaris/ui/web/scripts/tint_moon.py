import os
from PIL import Image, ImageEnhance
import math

def create_maps():
    print("Loading 8k_moon.jpg...")
    img = Image.open('../public/textures/8k_moon.jpg').convert('RGB')
    
    # We will resize to 4K to avoid memory issues and make loading faster on the web
    print("Resizing to 4K...")
    img = img.resize((4096, 2048), Image.Resampling.LANCZOS)
    width, height = img.size
    
    # Increase base contrast slightly for a punchier look
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    pixels = img.load()
    
    aesthetic_img = Image.new('RGB', (width, height))
    a_pixels = aesthetic_img.load()
    
    gravity_img = Image.new('RGB', (width, height))
    g_pixels = gravity_img.load()
    
    print("Processing pixels for Aesthetic & Gravity maps...")
    for y in range(height):
        # Normalize y coordinate -1 to 1 (latitude)
        ny = (y / height) * 2 - 1
        for x in range(width):
            # Normalize x coordinate -1 to 1 (longitude)
            nx = (x / width) * 2 - 1
            
            r, g, b = pixels[x, y]
            intensity = (r + g + b) / 3.0
            
            # --- Aesthetic False Color (Galileo Style) ---
            # We want beautiful gray highlands and highly saturated but subtle blue/orange maria.
            if intensity < 140:
                # Use spatial coordinates to create smooth natural-looking patches
                noise1 = math.sin(nx * 10) * math.cos(ny * 10)
                noise2 = math.sin(nx * 20 + ny * 5)
                patch = noise1 + noise2 * 0.5
                
                # Base tint
                tr, tg, tb = r, g, b
                if patch > 0.3:
                    # Blue (High Titanium basalt)
                    tr = int(r * 0.8)
                    tg = int(g * 0.9)
                    tb = int(min(255, b * 1.6))
                elif patch < -0.3:
                    # Orange/Red (Low Titanium basalt)
                    tr = int(min(255, r * 1.5))
                    tg = int(g * 1.1)
                    tb = int(b * 0.8)
                else:
                    # Transitional pinkish/purple
                    tr = int(min(255, r * 1.2))
                    tg = int(g * 0.9)
                    tb = int(min(255, b * 1.2))
                
                # Blend factor depends on how dark it is (darker = stronger color)
                blend = min(0.8, (140 - intensity) / 100.0)
                a_pixels[x, y] = (
                    int(r * (1 - blend) + tr * blend),
                    int(g * (1 - blend) + tg * blend),
                    int(b * (1 - blend) + tb * blend)
                )
            else:
                # Highlands: Very bright and slightly warmer, high contrast
                bright_r = min(255, int(r * 1.05))
                bright_g = min(255, int(g * 1.02))
                bright_b = min(255, int(b * 0.98))
                a_pixels[x, y] = (bright_r, bright_g, bright_b)
                
            # --- Gravity Anomaly Map (Scientific Turbo/Rainbow Map) ---
            norm = intensity / 255.0
            
            # Scientific 'Turbo' approximation (smooth blue -> cyan -> green -> yellow -> red)
            # Dark areas (low elevation) = Blue/Cyan
            # Bright areas (high elevation) = Red/Yellow
            t_r = max(0, min(255, int(255 * (3.0 * norm - 1.5))))
            if norm < 0.5:
                t_g = max(0, min(255, int(255 * (2.0 * norm))))
            else:
                t_g = max(0, min(255, int(255 * (2.0 - 2.0 * norm))))
            t_b = max(0, min(255, int(255 * (1.5 - 3.0 * norm))))
            
            # Blend heavily with the original texture so craters are still visible
            g_pixels[x, y] = (
                int(t_r * 0.7 + r * 0.3),
                int(t_g * 0.7 + g * 0.3),
                int(t_b * 0.7 + b * 0.3)
            )

    print("Saving aesthetic_moon_real.jpg...")
    aesthetic_img.save('../public/textures/aesthetic_moon_real.jpg', quality=90)
    
    print("Saving gravity_moon_real.jpg...")
    gravity_img.save('../public/textures/gravity_moon_real.jpg', quality=90)
    
    print("Done!")

if __name__ == '__main__':
    create_maps()
